from torch.cuda.amp import autocast
import argparse
import datetime
import functools
import os
from os.path import join
import random
from jax._src.api import mask

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset.helpers import collate_fn
from dataset.dataset import AttentiveDistDataset
from dataset.openfold_util import residue_constants

from model.alphafold_finetune import AlphaFold
import util2
import rmsd
from model.openfold.feats import atom14_to_atom37
from dataset.openfold_util import protein
from model.openfold.import_weights_partial import import_jax_weights_
host = os.uname()[1]
if "gilbreth" in host:
    RENUM_CHAIN_DIR = "/scratch/gilbreth/verburgt/training_data/renum_chain_dir"
else:
    RENUM_CHAIN_DIR = "/net/kihara/scratch/zhan1797/val2/new_test_pdb"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_dataset(args, datatype):
    targets_file = args.test_targets if datatype == "test" else args.val_targets if datatype == "val" else args.train_targets
    feature_dir = args.test_feature_dir if datatype == "test" else args.train_feature_dir
    label_dir = args.test_label_dir if datatype == "test" else args.train_label_dir
    seq_dir = args.test_seq_dir if datatype == "test" else args.train_seq_dir
    template_dir = args.test_template_dir if datatype == "test" else args.train_template_dir

    if args.gpcr:
        return MultimerDataset(args)
    else:
        return AttentiveDistDataset(
            targets_file=targets_file,
            feature_dir=feature_dir,
            label_dir=label_dir,
            seq_dir=seq_dir,
            max_len=args.max_len,
            datatype=datatype,
            use_templates=args.use_templates,
            template_dir=template_dir,
            template_chain_dir=args.template_chain_dir,
            train_size=args.train_size,
            embed=args.embed,
            full=args.full,
            msa_embedding_dir=args.msa_transformer_dir,
            af2=args.af2,
            msa=args.msa,
            e2e=args.e2e,
            hhbond=args.hhbond,
            args=args,
        )



def train(args, model, train_dataloader, val_dataloader, test_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    if args.model_dir:  
        if args.ignore_lddt_model:  # JAKE partially loading from a previous model
            print("Loading only non-plddt models")
            params = torch.load(f'{args.model_dir}/model_state_dict.pt', map_location='cpu')
            new_keys = {}
            for key in model.state_dict().keys():
                if key.startswith("plddt") or key.startswith("mqa_feat_transition") or key.startswith("mqa_edgefeat_transition") or key.startswith("esm_groupconv"):  #Don't load these ones because they're the wrong size for new network
                    new_keys[key] = model.state_dict()[key]  #Just keep the default value
                    continue
                new_keys[key] = params[key]  
            model.load_state_dict(new_keys)

        else: #Load the whole model as per Zicong's code
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(
                torch.load(f'{args.model_dir}/model_state_dict.pt',
                        map_location='cpu')
            )
            model.cuda(args.device_id)
            #Loading the optimizer
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            optimizer.load_state_dict(
                torch.load(f'{args.model_dir}/optimizer.pt',
                        map_location=f'{device}:{args.device_id}')
            )
            # Added for restart in torch 1.12+  sometimes needed depending on how model is loaded
            optimizer.param_groups[0]['capturable'] = True #Need this of loading whole model
            print(
                f'Checkpoints (model and optimizer) loaded from {args.model_dir}')
            
        # print("saving model to file")
        # torch.save(model.cpu().state_dict(), "/home/kihara/verburgt/alphafold/af2_graphqa/models_for_testing/gat_fromtrain.pt" )
        # exit()

        if args.freeze_non_lddt:  #Only train LDDT Network params
            #Freeze all the params
            print("Jake - Freezing Non-LDDT model parameters")
            for param in model.parameters():
                param.requires_grad = False

            #Unfreeze only the plddt network params
            for param in model.plddt.parameters():
                param.requires_grad = True

            #Also unfreeze mqa tranistion layers if you are using
            if model.mqa_transition_needed:
                print("Unfreezing mqa_transition (extra features) layers")
                for param in model.mqa_feat_transition.parameters():  #33 + 384 --> 384
                    param.requires_grad = True

            if model.edges_needed:
                print("Unfreezing mqa_edgefeat_transition (edge features) layers")
                for param in model.mqa_edgefeat_transition.parameters():  #33 + 384 --> 384
                    param.requires_grad = True

            #If you are using the full ESM attentions, unfreeze the transition layer
            if args.esm_edgefeats_alllayer:
                print("Unfreezing esm attention groupconvolution layer")
                for param in model.esm_groupconv.parameters(): 
                    param.requires_grad = True
    
            #Unfreeze transition params if flag set
            if args.unfreeze_transition:
                print("unfreezing transition layers")
                for param in model.structure_module.transition.parameters():
                    param.requires_grad = True

        # Added for restart in torch 1.12+  sometimes needed depending on how model is loaded
        # try:
        #     optimizer.param_groups[0]['capturable'] = True
        # except Exception as e:
        #     print("'optimizer.param_groups[0]['capturable'] = True' Failed!")  #Don't hard fail if this doesn't work

    # Scheduler
    constant_lr_epochs = args.constant_lr_epochs

    criterion_msa = torch.nn.CrossEntropyLoss()
    tb_output_dir = os.path.join(args.output_dir, "tensorboard")
    tb_writer = SummaryWriter(tb_output_dir)
    out_file = open(os.path.join(args.output_dir, "log"), 'w')
    print("----------------- Starting Training ---------------")
    print("  Num examples = %d" % (int(args.training_examples)))
    print("  Num Epochs = %d" % (int(args.epochs)))
    print("  Batch Size = %d" % (int(args.batch)))
    print("  Tensorboard directory = %s" % (tb_output_dir))

    # print("DUMPING PARAM DATA - JAKE")
    # for name, parameter in model.named_parameters():
    #     print(name, parameter.requires_grad)
    # exit()

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.train()
    model.zero_grad()

    #Make CSV
    lddt_error_csv = os.path.join(args.output_dir, "LDDT_Errors.csv")
    lddt_error_df = pd.DataFrame(columns=["global_step", "target", "lddt_error", "plddt_loss_unweighted", "plddt_loss"])

        
    if args.val_only:
        #casp_results = test(args, model, test_dataloader)
        results = val(args, model, val_dataloader)

        for key, value in results.items():
            if key not in ['target', "target_list", "lddt_errors",]:
                tb_writer.add_scalar("val_{}".format(key), value, global_step)
            if key == "lddt_errors":
                pass

        lddt_errors = results["lddt_errors"].cpu().numpy()
        target_list = results["target_list"]
        new_values_df = pd.DataFrame(list(zip(lddt_errors, target_list)), columns = ["lddt_error", "target"]) 
        new_values_df["global_step"] = global_step
        new_values_df["plddt_loss_unweighted"] = float(results["plddt_loss_unweighted"])
        new_values_df["plddt_loss"] = float(results["plddt loss"])
        lddt_error_df = lddt_error_df.append(new_values_df, ignore_index = True)
        lddt_error_df.to_csv(lddt_error_csv, index = False)
        print("exiting after initial validation...")
        exit()

    gt_keys = ['all_atom_positions', 'all_atom_mask', 'atom14_atom_exists', 'atom14_gt_exists', 'atom14_gt_positions',
               'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists',
               'atom14_atom_is_ambiguous', 'residue_index']
    gt_frames_keys = ['rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_alt_gt_frames',
                      'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask', 'chi_angles_sin_cos', 'chi_mask', 'seq_mask']

    for epoch in range(args.epochs):
        for step, (batch, targets) in enumerate(train_dataloader):
            print("step is", step)
            print("targets are", targets) #JAKE

            # for key, val in batch.items():
            #     if isinstance(val, torch.Tensor):
            #         print(key, val.shape)
            # exit()
            # with autograd.detect_anomaly():
            model.train()
            embedding = batch['embed']
            single_repr_batch = batch['single_representation']
            #coords_batch = batch["coords"]
            #masks_batch = batch["mask"].bool()
            aatype_batch = batch["aatype"]
            batch_gt = {key: batch[key] for key in gt_keys}
            batch_gt_frames = {key: batch[key] for key in gt_frames_keys}

            batch_gt.update({'seq_length': batch['seq_length']})
            resolution = batch['resolution']
            representation = None

            batch_gt.update({'pair_missing': batch['pair_missing']})
            batch_gt.update({'missing': batch['missing']})
            batch_gt.update({'single_missing': batch['single_missing']})
            batch_gt.update({'aatype_missing': batch['aatype_missing']})
            batch_gt.update({'seq_mask_missing': batch['seq_mask_missing']})

            # #Add targets to batch - Jake
            # batch_gt.update({'targets': batch['targets']})
            #torch.save(targets, "/home/verburgt/TEST_TARGETS.pt")

            if args.esm_feats:
                batch_gt.update({'esm': batch['esm']})
            if args.rmsf_feats:
                batch_gt.update({'rmsf': batch['rmsf']})
            if args.esm_edgefeats:
                batch_gt.update({'esm_edge': batch['esm_edge']})
            if args.esm_edgefeats_lastlayer:
                batch_gt.update({'esm_lastedge': batch['esm_lastedge']})
            if args.esm_edgefeats_alllayer:
                batch_gt.update({'esm_alledge': batch['esm_alledge']})


            if args.msa:
                bert_mask_ = batch['msa_mask'][0].unsqueeze(0)
                bert_mask, true_msa = batch['msa_mask'][0].unsqueeze(
                    0).bool(), batch['msa_mask'][0].unsqueeze(0)
            if args.use_templates:
                template1 = batch['template1']
                template2 = batch['template2']
                template3 = batch['template3']
                template4 = batch['template4']
                template5 = batch['template5']
            if args.e2e:
                representation = batch['representation']
                batch_gt.update({  # 'msa_mask': batch['msa_mask'],
                                'pseudo_beta': batch['pseudo_beta'],
                                'pseudo_beta_mask': batch['pseudo_beta_mask']})

            if args.cuda:
                embedding = embedding.cuda(args.device_id)
                if args.e2e:
                    representation = representation.cuda(args.device_id)

                resolution = resolution.cuda(args.device_id)
                for key in batch_gt.keys():
                    batch_gt[key] = batch_gt[key].cuda(args.device_id)
                for key in batch_gt_frames.keys():
                    batch_gt_frames[key] = batch_gt_frames[key].cuda(
                        args.device_id)
                single_repr_batch = single_repr_batch.cuda(args.device_id)
                #coords_batch = coords_batch.cuda(args.device_id)
                #masks_batch = masks_batch.cuda(args.device_id)
                aatype_batch = aatype_batch.cuda(args.device_id)
                #trans_msa = trans_msa.cuda(args.device_id)
                #trans_pair = trans_pair.cuda(args.device_id)
                if args.msa:
                    bert_mask_ = bert_mask_.cuda(args.device_id)
                    bert_mask = bert_mask.cuda(args.device_id)
                    true_msa = true_msa.cuda(args.device_id)
                if args.use_templates:
                    template1 = template1.cuda(args.device_id)
                    template2 = template2.cuda(args.device_id)
                    template3 = template3.cuda(args.device_id)
                    template4 = template4.cuda(args.device_id)
                    template5 = template5.cuda(args.device_id)
                else:
                    template1, template2, template3, template4, template5 = None, None, None, None, None

                if args.MT:
                    # mt_single = batch['mt_single'].cuda(args.device_id)
                    # mt_pair = batch['mt_pair'].cuda(args.device_id)
                    # mt_msa = batch['mt_msa'].cuda(args.device_id)
                    embedding2 = batch['embed2'].cuda(args.device_id)

                if args.hhbond:
                    batch_gt.update(
                        {'hhbond': batch['hhbond'].cuda(args.device_id)})
                    batch_gt.update(
                        {'hydro': batch['hydro'].cuda(args.device_id)})
                    batch_gt.update(
                        {'ionic': batch['ionic'].cuda(args.device_id)})

                if args.distill:
                    batch_gt.update(
                        {'embed_distill': batch['embed_distill'].cuda(args.device_id)})
                    batch_gt.update(
                        {'single_distill': batch['single_distill'].cuda(args.device_id)})

            # autocast
            # with autocast():
            if args.MT:
                outputs2 = model(embedding, single_repr_batch, aatype_batch, batch_gt, batch_gt_frames,
                                 resolution, representation=representation, emb2=embedding2)
            else:
                outputs2 = model(embedding, single_repr_batch, aatype_batch, batch_gt, batch_gt_frames, resolution,
                                 representation=representation)
            
            # print("MODEL_RAN, EXITING")
            # exit()
            loss2 = outputs2[1]
            violation_loss, angle_loss, plddt_loss = outputs2[3], outputs2[4], outputs2[5]
            bbloss, scloss = outputs2[-5], outputs2[-4]
            if args.e2e:
                dist_loss = outputs2[6]
            if args.hhbond:
                hh_loss = outputs2[-3]
                hydro_loss = outputs2[-2]
                ionic_loss = outputs2[-1]
            # print(loss2)
            if torch.sum(torch.isnan(loss2)) == 1:
                print("targets are:", targets)
                model.zero_grad()
                continue

            loss = loss2
            # print(loss2)

            if args.msa:
                '''
                true_msa = rearrange(true_msa.squeeze(0)*bert_mask_.squeeze(0), 'h w -> (h w)').long()
                msa = rearrange(msa.squeeze(0)*bert_mask_.squeeze(0)[..., None], 's r c -> (s r) c')
                '''
                # print(msa.size())
                # print(true_msa.size())
                msa = msa*bert_mask[..., None]
                msa = msa.permute(0, 3, 1, 2)
                msa_loss = criterion_msa(msa, (true_msa*bert_mask).long())
                loss = loss + msa_loss * 2
                #print('msa loss:', msa_loss)
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                tr_loss += loss.item()
                optimizer.step()
                global_step += 1

                total_norm = 0
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        param_norm = parameter.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                optimizer.zero_grad()
                model.zero_grad()
                # print(total_norm)
                tb_writer.add_scalar("gradient norm", total_norm, global_step)
                if args.e2e:
                    tb_writer.add_scalar("dist", dist_loss, global_step)
                tb_writer.add_scalar("loss2", loss2, global_step)
                tb_writer.add_scalar(
                    "violation loss", violation_loss, global_step)
                tb_writer.add_scalar("angle loss", angle_loss, global_step)
                tb_writer.add_scalar("plddt loss", plddt_loss, global_step)
                tb_writer.add_scalar("backbone loss", bbloss, global_step)
                tb_writer.add_scalar("sidechain loss", scloss, global_step)
                if args.hhbond:
                    tb_writer.add_scalar("hhbond loss", hh_loss, global_step)
                    tb_writer.add_scalar("hydro loss", hydro_loss, global_step)
                    tb_writer.add_scalar("ionic loss", ionic_loss, global_step)
                # Logging
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar(
                        "loss", (tr_loss - logging_loss) / args.logging_steps, global_step,)
                    tb_writer.add_scalar(
                        "lr", optimizer.param_groups[0]["lr"], global_step)
                    print("Epoch %s Global Step %s Loss %f" % (epoch, global_step, (tr_loss -
                          logging_loss) / args.logging_steps,), str(datetime.datetime.now()))
                    out_file.write("Epoch %s Global Step %s Loss %f \n" % (
                        epoch, global_step, (tr_loss - logging_loss) / args.logging_steps,))
                    out_file.flush()
                    logging_loss = tr_loss
            # print("done with one batch")
            # exit()
        # print("got through one epoch")
        # exit()
        # if args.val_logging_steps > 0 and global_step % args.val_logging_steps == 0:
        if args.val_epochs > 0 and epoch % args.val_epochs == 0 and epoch > 0:
            results = val(args, model, val_dataloader)
            for key, value in results.items():
                if key not in ['target', "target_list", "lddt_errors",]:
                    tb_writer.add_scalar("val_{}".format(key), value, global_step)
                if key == "lddt_errors":
                    pass
                    # mean_lddt_error = value.mean()
                    # tb_writer.add_scalar(
                    #     "val_{}".format(key), mean_lddt_error, global_step)

            lddt_errors = results["lddt_errors"].cpu().numpy()
            target_list = results["target_list"]
            new_values_df = pd.DataFrame(list(zip(lddt_errors, target_list)), columns = ["lddt_error", "target"]) 
            new_values_df["global_step"] = global_step
            new_values_df["plddt_loss_unweighted"] = float(results["plddt_loss_unweighted"])
            new_values_df["plddt_loss"] = float(results["plddt loss"])
            lddt_error_df = lddt_error_df.append(new_values_df, ignore_index = True)
            lddt_error_df.to_csv(lddt_error_csv, index = False)

            print("Eval", results, str(datetime.datetime.now()))
            out_file.write("Eval " + str(results)+"\n")
            out_file.flush()
            PREFIX_CHECKPOINT_DIR = "checkpoint"
            output_dir = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{epoch}-{global_step}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(model.state_dict(), os.path.join(
                output_dir, "model_state_dict.pt"),)
            torch.save(optimizer.state_dict(), os.path.join(
                output_dir, "optimizer.pt"))
            print("Saving model checkpoint to %s" % (output_dir))

    out_file.close()

def val(args, model, val_dataloader):

    eval_loss, eval_loss2 = 0.0, 0.0
    eval_steps, eval_steps2 = 0, 0
    vio, angle, plddt = 0, 0, 0
    bbloss, scloss = 0, 0
    distogram = 0
    #Keep track of unweighted plddt and lddt_error
    lddt_errors = []
    plddt_unweighted = 0
    target_list = []

    accuracies = {}
    model.eval()

    for step, (batch, targets) in enumerate(val_dataloader):
        #print("targets are ", targets)
        target_list.extend(targets)

        embedding = batch['embed']
        # if args.embed =='msa_transformer':
        #	represenation = batch['representation']
        representation = None
        if args.e2e:
            representation = batch['representation']

        gt_keys = ['all_atom_positions', 'all_atom_mask', 'atom14_atom_exists', 'atom14_gt_exists', 'atom14_gt_positions',
                   'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists',
                   'atom14_atom_is_ambiguous', 'residue_index']
        gt_frames_keys = ['rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_alt_gt_frames',
                          'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask',  'chi_angles_sin_cos', 'chi_mask', 'seq_mask']

        single_repr = batch['single_representation']
        #T_true = rigid_from_three_points(coords[:,:,0,:],coords[:,:,1,:],coords[:,:,2,:])
        records = batch['res_dict']["record_lines"]

        resolution = batch['resolution']
        aatype = batch['aatype']
        batch_gt = {key: batch[key] for key in gt_keys}
        batch_gt_frames = {key: batch[key] for key in gt_frames_keys}
        batch_gt.update({'seq_length': batch['seq_length']})

        batch_gt.update({'pair_missing': batch['pair_missing']})
        batch_gt.update({'missing': batch['missing']})
        batch_gt.update({'single_missing': batch['single_missing']})
        batch_gt.update({'aatype_missing': batch['aatype_missing']})
        batch_gt.update({'seq_mask_missing': batch['seq_mask_missing']})

        if args.esm_feats:
            batch_gt.update({'esm': batch['esm']})
        if args.rmsf_feats:
            batch_gt.update({'rmsf': batch['rmsf']})
        if args.esm_edgefeats:
            batch_gt.update({'esm_edge': batch['esm_edge']})
        if args.esm_edgefeats_lastlayer:
            batch_gt.update({'esm_lastedge': batch['esm_lastedge']})
        if args.esm_edgefeats_alllayer:
            batch_gt.update({'esm_alledge': batch['esm_alledge']})

        if args.msa:
            bert_mask_ = batch['msa_mask'][0].unsqueeze(0)
            bert_mask, true_msa = batch['msa_mask'][0].unsqueeze(
                0).bool(), batch['msa_mask'][0].unsqueeze(0)
        if args.use_templates:
            template1 = batch['template1']
            template2 = batch['template2']
            template3 = batch['template3']
            template4 = batch['template4']
            template5 = batch['template5']

        if args.cuda:
            embedding = embedding.cuda(args.device_id)
            if args.e2e:
                representation = representation.cuda(args.device_id)
                batch_gt.update({  # 'msa_mask': batch['msa_mask'],
                                'pseudo_beta': batch['pseudo_beta'],
                                'pseudo_beta_mask': batch['pseudo_beta_mask']})

            single_repr = single_repr.cuda(args.device_id)

            aatype = aatype.cuda(args.device_id)
            resolution = resolution.cuda(args.device_id)
            for key in batch_gt.keys():
                batch_gt[key] = batch_gt[key].cuda(args.device_id)
            for key in batch_gt_frames.keys():
                batch_gt_frames[key] = batch_gt_frames[key].cuda(
                    args.device_id)
            if args.msa:
                bert_mask_ = bert_mask_.cuda(args.device_id)
                bert_mask = bert_mask.cuda(args.device_id)
                true_msa = true_msa.cuda(args.device_id)
            if args.use_templates:
                template1 = template1.cuda(args.device_id)
                template2 = template2.cuda(args.device_id)
                template3 = template3.cuda(args.device_id)
                template4 = template4.cuda(args.device_id)
                template5 = template5.cuda(args.device_id)
            else:
                template1, template2, template3, template4, template5 = None, None, None, None, None

            if args.MT:
                # mt_single = batch['mt_single'].cuda(args.device_id)
                # mt_pair = batch['mt_pair'].cuda(args.device_id)
                # mt_msa = batch['mt_msa'].cuda(args.device_id)
                embedding2 = batch['embed2'].cuda(args.device_id)
            if args.hhbond:
                batch_gt.update(
                    {'hhbond': batch['hhbond'].cuda(args.device_id)})
                batch_gt.update({'hydro': batch['hydro'].cuda(args.device_id)})
                batch_gt.update({'ionic': batch['ionic'].cuda(args.device_id)})

            if args.distill:
                batch_gt.update(
                    {'embed_distill': batch['embed_distill'].cuda(args.device_id)})
                batch_gt.update(
                    {'single_distill': batch['single_distill'].cuda(args.device_id)})
        #Runs in forward_training
        with torch.no_grad():
            # try:
            if args.MT:
                outputs2 = model(embedding, single_repr, aatype, batch_gt, batch_gt_frames,
                                 resolution, representation=representation, emb2=embedding2, 
                                 return_weighted_and_unweighted_plddt_loss = True)
            else:
                outputs2 = model(embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution,
                                 representation=representation, 
                                 return_weighted_and_unweighted_plddt_loss = True)
            # except:
            #     print('='*80)
            #     print(targets)
            #     print('='*80)
            #     continue
                
            # print("LDDT Stuff is", outputs2[5][3])
            # print("VAL MODEL RAN, EXITING")
            # exit()
            loss2 = outputs2[1]
            loss = loss2
            # print(loss)
            eval_loss += loss.item()
            vio += outputs2[3].item()
            angle += outputs2[4].item()
            plddt += outputs2[5][0].item() #Adjusted for return_weighted_and_unweighted_plddt_loss
            plddt_unweighted += outputs2[5][1].item() #Adjusted for return_weighted_and_unweighted_plddt_loss
            lddt_errors.append(outputs2[5][3]) #Adjusted for return_weighted_and_unweighted_plddt_loss
            bbloss += outputs2[-5].item()
            scloss += outputs2[-4].item()
            if args.e2e:
                distogram += outputs2[6].item()
        # now here it comes
        final_pos = atom14_to_atom37(outputs2[2][-1], batch_gt)
        # print(final_pos.size())
        final_atom_mask = batch_gt["atom37_atom_exists"]
        # print(final_atom_mask.size())
        pdb_dir = RENUM_CHAIN_DIR

        # Ground Truth PDB
        pdb = join(pdb_dir, targets[0]+'.pdb')
        # print(pdb)
        pdb_str = ''
        with open(pdb, "r") as f:
            pdb_str = f.read()
        prot_true = protein.from_pdb_string(pdb_str)
        pdb_lines = protein.to_pdb(prot_true)
        output_dir_true = os.path.join(
            args.output_dir, f"{targets[0]}_true_all.pdb")
        with open(output_dir_true, 'w') as f:
            f.write(pdb_lines)

        # Load B Factor with plddt prediction #JAKE
        #############################################################################################
        # This code was ripped from /net/kihara/home/zhan1797/Desktop/casp15/pred_test.py for quick
        # addition of predicted plddt to validation output. Will better implement/optimize later - Jake
        # with torch.no_grad():  #Runs in forward_testing
        #     #print("batch_gt type is:", batch_gt)
        #     _, postition_full, old_confidence = model(embedding, single_repr, aatype,
        #                                           batch_gt, None, None, training=False,
        #                                           representation=representation)

        confidence = outputs2[5][2] # B X max_len

        #print("Confidence is", confidence)
        confidence = confidence.cpu().numpy()
        #print(confidence)
        plddt_b_factors = np.repeat(
            confidence[..., None], residue_constants.atom_type_num, axis=-1
        )
        plddt_b_factors = plddt_b_factors.reshape(-1, 37)

        #print("plddt_b_factors is", plddt_b_factors)

        ################################################################################################
        prot = protein.Protein(
            aatype=aatype.squeeze(0).cpu().numpy(),
            atom_positions=final_pos.squeeze(0).cpu().numpy(),
            atom_mask=final_atom_mask.squeeze(0).cpu().numpy(),
            residue_index=prot_true.residue_index,
            b_factors=plddt_b_factors,
        )  # Took out the plus 1 to fix indexing and added b factors
        pdb_lines = protein.to_pdb(prot)
        # print(len(pdb_lines))
        output_dir_pred = os.path.join(
            args.output_dir, f"{targets[0]}_pred_all.pdb")
        with open(output_dir_pred, 'w') as f:
            f.write(pdb_lines)

        try:
            # this example doesnt have N for the first amino acid in the pdb file
            if targets[0] == '7C7QB':
                prot_true = protein.Protein(
                    aatype=prot_true.aatype[1:],
                    atom_positions=prot_true.atom_positions[1:, ...],
                    atom_mask=prot_true.atom_mask[1:, ...],
                    residue_index=prot_true.residue_index[1:],
                    b_factors=prot_true.b_factors[1:],
                )
            # calculate RMSD
            gt = prot_true.atom_positions[None, ...][prot_true.atom_mask[None, ...].astype(
                bool)]
            #pred = outputs2[0][masks].unsqueeze(0).cpu().detach().numpy()
            pred = prot.atom_positions[None, ...][prot_true.atom_mask[None, ...].astype(
                bool)]
            gt = gt.reshape(-1, 3)
            pred = pred.reshape(-1, 3)
            gt -= rmsd.centroid(gt)
            pred -= rmsd.centroid(pred)
            U = rmsd.kabsch(gt, pred)
            A = np.dot(gt, U)
            value = rmsd.rmsd(A, pred)
            eval_steps2 += 1
        except:
            value = 0
        eval_loss2 += value
        eval_steps += 1
    #End of val loop
    avg_eval_loss = eval_loss / eval_steps
    avg_angle = angle / eval_steps
    avg_vio = vio / eval_steps
    avg_plddt = plddt / eval_steps
    avg_plddt_unweighted = plddt_unweighted / eval_steps
    avg_dist = distogram / eval_steps
    avg_bb = bbloss / eval_steps
    avg_sc = scloss / eval_steps
    if eval_steps2 == 0:
        avg_rmsd = 0
    else:
        avg_rmsd = eval_loss2 / eval_steps2

    #Average lddt error
    lddt_errors = torch.cat(lddt_errors)
    # print("lddt_errors are", lddt_errors)
    # print("targets_are", target_list)
    # print("VAL RAN, EXITING")
    # exit()
    #result = {"loss": avg_eval_loss, "rmsd": avg_rmsd}
    result = {"loss": avg_eval_loss, "rmsd": avg_rmsd, 'violation': avg_vio, 'angle loss': avg_angle, 'plddt loss': avg_plddt, 'dist loss': avg_dist,
              'backbone': avg_bb, 'sidechain': avg_sc, 'targets': eval_steps2, "plddt_loss_unweighted": avg_plddt_unweighted, "lddt_errors": lddt_errors, 
              "target_list":target_list}
    return result



def test(args, model,test_dataloader):
    if args.model_dir:  
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(
            torch.load(f'{args.model_dir}/model_state_dict.pt',
                    map_location='cpu')
        )
        model.cuda(args.device_id)
    else:
        print("Model directory must be set for testing!")
        raise Exception()


    model.eval()
    for step, (batch, targets) in enumerate(test_dataloader):
        embedding = batch['embed']
        representation = None
        if args.e2e:
            representation = batch['representation']

        gt_keys = ['all_atom_positions', 'all_atom_mask', 'atom14_atom_exists', 'atom14_gt_exists', 'atom14_gt_positions',
                   'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists',
                   'atom14_atom_is_ambiguous', 'residue_index']
        gt_frames_keys = ['rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_alt_gt_frames',
                          'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask',  'chi_angles_sin_cos', 'chi_mask', 'seq_mask']

        single_repr = batch['single_representation']
        #T_true = rigid_from_three_points(coords[:,:,0,:],coords[:,:,1,:],coords[:,:,2,:])
        records = batch['res_dict']["record_lines"]

        resolution = batch['resolution']
        aatype = batch['aatype']
        batch_gt = {key: batch[key] for key in gt_keys}
        batch_gt_frames = {key: batch[key] for key in gt_frames_keys}
        batch_gt.update({'seq_length': batch['seq_length']})

        batch_gt.update({'pair_missing': batch['pair_missing']})
        batch_gt.update({'missing': batch['missing']})
        batch_gt.update({'single_missing': batch['single_missing']})
        batch_gt.update({'aatype_missing': batch['aatype_missing']})
        batch_gt.update({'seq_mask_missing': batch['seq_mask_missing']})
        
        #batch_gt.update({'missing': torch.ones_like(batch['missing'])})

        # torch.save( batch_gt["seq_mask_missing"], "TEMP_batch_gt_seq_mask_missing.pt")
        # torch.save(batch_gt["aatype_missing"], "TEMP_batch_gt_aatype_missing.pt")
        # torch.save( batch_gt['single_missing'], "TEMP_batch_gt_single_missing.pt")
        # torch.save( batch_gt['missing'], "TEMP_batch_gt_missing.pt")

        if args.esm_feats:
            batch_gt.update({'esm': batch['esm']})
        if args.rmsf_feats:
            batch_gt.update({'rmsf': batch['rmsf']})
        if args.esm_edgefeats:
            batch_gt.update({'esm_edge': batch['esm_edge']})
        if args.esm_edgefeats_lastlayer:
            batch_gt.update({'esm_lastedge': batch['esm_lastedge']})
        if args.esm_edgefeats_alllayer:
            batch_gt.update({'esm_alledge': batch['esm_alledge']})


        if args.cuda:
            embedding = embedding.cuda(args.device_id)
            if args.e2e:
                representation = representation.cuda(args.device_id)
                batch_gt.update({  # 'msa_mask': batch['msa_mask'],
                                'pseudo_beta': batch['pseudo_beta'],
                                'pseudo_beta_mask': batch['pseudo_beta_mask']})

            single_repr = single_repr.cuda(args.device_id)

            aatype = aatype.cuda(args.device_id)
            #print("AAtype is", aatype)
            resolution = resolution.cuda(args.device_id)
            for key in batch_gt.keys():
                batch_gt[key] = batch_gt[key].cuda(args.device_id)
            for key in batch_gt_frames.keys():
                batch_gt_frames[key] = batch_gt_frames[key].cuda(
                    args.device_id)


            if args.distill:
                batch_gt.update(
                    {'embed_distill': batch['embed_distill'].cuda(args.device_id)})
                batch_gt.update(
                    {'single_distill': batch['single_distill'].cuda(args.device_id)})
        #Runs in forward_testing
        with torch.no_grad():
            outputs2 = model(embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution,
                                representation=representation, 
                                return_weighted_and_unweighted_plddt_loss = True)

        representation_test = None
        single_test = batch['test']['single']
        pair_test = batch['test']['pair']
        aatype_test = batch['test']['aatype']
        protein_test = batch['test']['protein']

        # torch.save(outputs2[2][-1].cpu(), "putputs_2_neg1.pt" )
        # torch.save(batch_gt, "batchgt.pt")
        # torch.save(batch, "batch.pt")
        final_pos = atom14_to_atom37(outputs2[0][-1].cpu(), protein_test)  # 2 --> 0
        final_atom_mask = protein_test["atom37_atom_exists"]

        confidence = outputs2[1] # B X max_len [5][2] --> 1
        confidence = confidence.cpu().numpy()
        plddt_b_factors = np.repeat(
            confidence[..., None], residue_constants.atom_type_num, axis=-1
        )
        plddt_b_factors = plddt_b_factors.reshape(-1, 37)
        prot = protein.Protein(
            aatype=aatype_test.squeeze(0).cpu().numpy(),
            atom_positions=final_pos.squeeze(0).cpu().numpy(),
            atom_mask=final_atom_mask.squeeze(0).cpu().numpy(),
            residue_index=protein_test['residue_index'].squeeze(0).numpy() + 1,
            b_factors=plddt_b_factors,
        )
        pdb_lines = protein.to_pdb(prot)
        output_dir_pred = os.path.join(
            args.output_dir, f"{targets[0]}_pred_all.pdb")
        with open(output_dir_pred, 'w') as f:
            f.write(pdb_lines)
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contact_cutoff', type=float, default=5.0,
                        help='Distance in Angstroms to detect a contact between Ca atoms when constructing the LDDT Graph')
    parser.add_argument('--graph_type', type=str, default="GCN", help='Type of graph to use (GAT or GCN)')
    parser.add_argument('--lddt_weight', type=float, default=1.0, help='Weight to multiply LDDT Loss by')
    parser.add_argument('--ignore_lddt_model', default=False, action="store_true", help='Will ignore lddt model params if set to true')
    parser.add_argument('--freeze_non_lddt', default=False, action="store_true", help='Will only train LDDT values if set to true')
    parser.add_argument('--unfreeze_transition', default=False, action="store_true", help='Will unfreeze transition layers if freeze_non_lddt is set')
    parser.add_argument('--graph_layers', type=int, default=4, help='Will ignore lddt model params if set to true')
    parser.add_argument('--lddt_weight_vector_file', type=str, default=None, help='pytorch file that has weights for all 50 bins')
    parser.add_argument('--esm_feats', default=False, action="store_true", help='whether or not to use additional features')
    parser.add_argument('--esm_edgefeats', default=False, action="store_true", help='whether or not to use additional features')
    parser.add_argument('--esm_edgefeats_lastlayer', default=False, action="store_true", help='whether or not to use additional features')
    parser.add_argument('--esm_edgefeats_alllayer', default=False, action="store_true", help='whether or not to use additional features')
    parser.add_argument('--rmsf_feats', default=False, action="store_true", help='whether or not to use rmsf features')
    parser.add_argument('--edge_feats', default=False, action="store_true", help='whether or not to use Edge features')
    parser.add_argument('--val_only', default=False, action="store_true", help='If true will only run initial validation')
    parser.add_argument('--test_only', default=False, action="store_true", help='If true will only Test')

    parser.add_argument('--aa_lddt', default=False, action="store_true", help='If true, All atom LDDT predictions will be made. False performs standard LDDT-CA predictions')

    parser.add_argument("--train_targets", default="/train_chains", type=str,
                        help="File of targets for training")
    parser.add_argument("--val_targets", default=None, type=str,
                        help="File of targets for val")
    parser.add_argument("--test_targets", default=None, type=str,
                        help="File of targets for test")
    parser.add_argument("--train_feature_dir", default="/net/kihara-fast-scratch/jain163/attentivedist2_data/inputs", type=str,
                        help="Directory containing train target features")
    parser.add_argument("--test_feature_dir", default="/net/kihara-fast-scratch/jain163/attentivedist2_data/inputs", type=str,
                        help="Directory containing test target features")
    parser.add_argument("--train_label_dir", default="/net/kihara-fast-scratch/jain163/attentivedist2_data/labels", type=str,
                        help="Directory containing train target labels")
    parser.add_argument("--test_label_dir", default="", type=str,
                        help="Directory containing eval target labels")
    parser.add_argument("--train_seq_dir", default="/net/kihara-fast-scratch/jain163/attentivedist2_data/seq", type=str,
                        help="Directory containing train target sequences")
    parser.add_argument("--test_seq_dir", default="/net/kihara-fast-scratch/jain163/attentivedist2_data/seq", type=str,
                        help="Directory containing eval target sequences")
    parser.add_argument("--train_template_dir", default="/net/kihara-fast-scratch/jain163/attentivedist2_data/template", type=str,
                        help="The directory containing templates information")
    parser.add_argument("--test_template_dir", default="/net/kihara-fast-scratch/jain163/attentivedist2_data/template", type=str,
                        help="The directory containing test templates information")
    parser.add_argument("--template_chain_dir", default="/net/kihara-fast-scratch/jain163/attentivedist2_data/template_chains_info", type=str,
                        help="The directory containing properties of each template chain")
    parser.add_argument("--output_dir", default="test_run", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_dir", default="", type=str,
                        help="model directory if load model from checkpoints")
    parser.add_argument("--msa_transformer_dir", default="/net/kihara-fast-scratch-2/zhan1797/attentivedist2_data/msa_transformer_embeddings", type=str,
                        help="The directory where pre-generated msa embeddings are stored.")

    parser.add_argument("--max_len", type=int, default=384,
                        help="Maximum sequnce length, larger proteins are clipped")
    parser.add_argument("--epochs", default=3600, type=int,
                        help="Total number of training epochs.")
    parser.add_argument('--num_blocks', type=int,
                        default=1, help='Number of 2d blocks')
    parser.add_argument('--num_attn_blocks', type=int,
                        default=1, help='Number of 2d blocks for attention')
    parser.add_argument('--channels', type=int,
                        default=64, help='Resnet channels')
    parser.add_argument('--dropout', type=float,
                        default=0.2, help='Dropout probability')
    parser.add_argument('--dilations', type=int, nargs='+',
                        default=[1, 2, 4], help='Cyclic dialations')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='L2 Regularization')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Num of workers in dataloader')
    parser.add_argument('--use_templates', default=False,
                        action="store_true", help='Use templates in input')
    parser.add_argument('--train_size', type=int, default=0,
                        help='Specify train datapoints to use, 0 means all')
    parser.add_argument('--model', type=str, default='alphafold',
                        help='Options: alphafold |resnet | resnest | saa')
    parser.add_argument('--resnest_blocks', type=int, nargs='+',
                        default=[2, 1, 1, 1], help='ResNeSt blocks, each block has 3 conv')
    parser.add_argument('--embed', type=str, default='msa_transformer',
                        help='Options: onehot | tape | onehot_tape | msa_transformer')
    parser.add_argument('--constant_lr_epochs', type=int,
                        default=30, help='Epochs before lr is decreased')

    parser.add_argument('--in_channels_msa', type=int,
                        default=105, help='Input channels for msa features')
    parser.add_argument('--in_channels_template', type=int,
                        default=64, help='Input channels for template features')
    parser.add_argument('--in_channels_embed', type=int, default=144,
                        help='Input channels for transformer embedding features')
    parser.add_argument('--out_channels_dist', type=int,
                        default=20, help='Distance output bins/channels')
    parser.add_argument('--out_channels_angle', type=int,
                        default=72, help='Backbone angle output bins/channels')
    parser.add_argument('--out_channels_mu', type=int,
                        default=25, help='Omega output bins/channels')
    parser.add_argument('--out_channels_theta', type=int,
                        default=25, help='Theta output bins/channels')
    parser.add_argument('--out_channels_rho', type=int,
                        default=13, help='Ori Phi output bins/channels')
    parser.add_argument('--out_channels_sce', type=int,
                        default=38, help='Sidechain center output bins/channels')
    parser.add_argument('--out_channels_no', type=int,
                        default=38, help='H-bond N-O output bins/channels')

    parser.add_argument("--logging_steps", type=int,
                        default=500, help="Log every X updates steps.")
    parser.add_argument("--val_logging_steps", type=int,
                        default=5000, help="Eval every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--val_epochs", type=int, default=50,
                        help="Save checkpoint every X updates steps.")

    parser.add_argument("--device_id", type=int,
                        default=0, help="cude device id")
    parser.add_argument("--seed", type=int, default=999,
                        help="random seed for initialization")

    parser.add_argument("--full", default=False, action="store_true",
                        help='use all four e-values for attentivedist')
    parser.add_argument("--af2", default=False,
                        action="store_true", help='whether use af2 embeddings')
    parser.add_argument("--e2e", default=False,
                        action="store_true", help='whether to train end to end')
    parser.add_argument("--msa", default=False,
                        action="store_true", help='whether to use msa loss')
    parser.add_argument("--recycle", default=False,
                        action="store_true", help='whether to use recycle')
    parser.add_argument("--ipa_depth", type=int, default=8,
                        help="depth of ipd block")
    parser.add_argument("--draw_epochs", type=int, default=1,
                        help="epochs for draw pdb files")
    parser.add_argument("--point_scale", type=int, default=10,
                        help="point scale for translations")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=4, help="gradient accumulation")
    parser.add_argument("--test", default=False, action="store_true",
                        help='whether to output sequence in full length')
    parser.add_argument("--gpcr", default=False, action="store_true",
                        help='whether to test on multimer targets')
    parser.add_argument("--MT", default=False, action="store_true",
                        help='whether to msa transformer embeddings')
    parser.add_argument("--hhbond", default=False,
                        action="store_true", help='whether to use hhbond loss')
    parser.add_argument("--distill", default=False,
                        action="store_true", help='whether to use distillation')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # Sets correct CUDA device order. https://github.com/matterport/Mask_RCNN/issues/109
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_properties(i).name)
    args.cuda = True if torch.cuda.is_available() else False
    args.n_gpu = 1  # Only use 1 gpu for now
    # Print and save args
    util2.print_options(args)
    # Set seed
    set_seed(args)

    collate = functools.partial(collate_fn, args=args)
    if args.test_only:
        args.test = True
        test_dataset = get_dataset(args, datatype="test")
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate)


    else:
        train_dataset = get_dataset(args, datatype="train")
        val_dataset = get_dataset(args, datatype="val")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch,
                                  shuffle=True, num_workers=args.num_workers, collate_fn=collate)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate)
        args.training_examples = len(train_dataset)
        exit()


    if args.lddt_weight_vector_file is None:
        args.lddt_weight_vector = torch.ones(50).cuda(args.device_id)
    else:
        args.lddt_weight_vector = torch.load(args.lddt_weight_vector_file).cuda(args.device_id)

    model = AlphaFold(args)
    if args.cuda:
        model.cuda(args.device_id)


    if args.test_only:
        print("Running Test")
        test(args, model,test_dataloader)
    else:
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            print(f'layer {name}: {param}')
            total_params += param
        print('trainable parameters: ', total_params)
        train(args, model, train_dataloader, val_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
