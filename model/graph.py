import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAT
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
import os

'''Classes and functions related to implementing pytorch_geometric graphs in the 
    Alphafold MQA prediction
    
    Jacob Verburgt (verburgt@purdue.edu)

'''

class mqa_feat_transition(torch.nn.Module):
    def __init__(self, in_features=3, out_features=16):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.lin1(x).relu()
        return x

class mqa_edgefeat_transition(torch.nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size = 1, padding=0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels, 
                                     kernel_size=kernel_size, 
                                     padding=padding)

    def forward(self, x):
        x = self.conv1(x).relu()
        return x

#V3
class PerResidueLDDTGraphPred(torch.nn.Module):
    '''GCN for better prediction of LDDT values. 
        Nodes should be L*384 and can be created with the "batch nodes" function from
        the single representation. 
        
        Edge_indexes are derived from contacts and can be created from the final position with the 
        "contact_edges" function '''
    def __init__(self, in_channels=384, out_channels=50,
                 gcn_channels=128, hidden_channels=128 
                 ):
        super().__init__()

        self.hidden_channels = hidden_channels     

        self.conv1 = GCNConv(in_channels, gcn_channels)
        self.conv2 = GCNConv(gcn_channels, gcn_channels)
        self.conv3 = GCNConv(gcn_channels, gcn_channels) #B*L x hidden_channels
        self.conv4 = GCNConv(gcn_channels, gcn_channels) #B*L x hidden_channels
        self.conv5 = GCNConv(gcn_channels, out_channels) #B*L x hidden_channels


    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]

        #Three GCN Layers
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        x = self.conv5(x, edge_index)

        return x
    

#Creating Edges
def create_contact_mask(missing:torch.Tensor, final_position) -> torch.Tensor:
    '''Creates a mask(Batch*max_len X Batch*max_len) to prevent edges being created from padded nodes. 
    Intended to be used as a mask for contact_edges()

    missing (Batch X L) - Binary matrix stored in batch_gt["missing"] . The largest should sum to max_len
    final_position (Batchxmax_lenx37atomsx3coord) - Used to get the batch size and max_length from
    '''
    #Pull shape from final position
    batch, size, _, _ = final_position.shape

    batch_lens = missing.sum(dim=1) #Get the length of unmasked residues in each batch. Maximum should be same as max_len
    mask_2D = torch.zeros(batch, size, size, device=final_position.device) #Create empty mask B X max_len X max_len
    for b in range(mask_2D.size(0)):         #Fill in the upper right corner with ones for each batch
        mask_2D[b, 0:int(batch_lens[b]), 0:int(batch_lens[b])] = 1

    mask_batch = torch.split(mask_2D, 1) #Split batches into tuples
    mask_batch = [t.reshape(size, size) for t in mask_batch] #Make them all LxL
    mask_diag  = torch.block_diag(*mask_batch) # Will be B*L x B*L
    return mask_diag


def create_full_contact_mask(missing:torch.Tensor) -> torch.Tensor:
    '''Creates a mask(Batch*L X Batch*L) to prevent edges being created from padded or missing nodes. 
    Intended to be used as a mask for contact_edges()
    missing (Batch X L) - Binary matrix stored in batch_gt["missing"] . The largest should sum to max_len
    '''
    L = missing.shape[1]
    mask2D = torch.einsum("bl,bk->blk", (missing, missing))  # B X L X L
    mask_batch = torch.split(mask2D, 1) #Split batches into tuples
    mask_batch = [t.reshape(L, L) for t in mask_batch] #Make them all LxL
    mask_diag  = torch.block_diag(*mask_batch) # Will be B*L x B*L
    return mask_diag


def contact_edges(final_position:torch.Tensor, contact:int = 5, mask:torch.Tensor = None) -> torch.Tensor:
    '''Takes in a "Final Position" tensor (BatchxLengthx37atomsx3coord) and returns (Here, Length is max_len)
    edge indexes (2 x num_edges) (which map source node index --> target node index)
    based on a distance ontact cutoff (default 5) in angstroms'''
    L = final_position.shape[1]
    coords = final_position[..., 1, :] #Extract CA coordinates (BxLx3)
    dists = torch.cdist(coords, coords) #calculate cross-distances (BxLxL output)
    conts =torch.where(dists < contact, 1, 0) #Use cuto/Contaff to convert to binary adacency matrix (still BxLxL)
    cont_batch = torch.split(conts, 1) #Split batches into tuples
    cont_batch = [t.reshape(L, L) for t in cont_batch] #Make them all LxL
    cont_all = torch.block_diag(*cont_batch) # Will be B*L x B*L

    if mask != None:
        assert cont_all.shape == mask.shape
        cont_all = cont_all * mask

    #torch.diagonal(cont_all, 0).zero_() #Zero diagnals
    edges = torch.argwhere(cont_all) #Contactsx2
    edge_index = edges.transpose(0,1) # Flip to 2xContacts
    return edge_index


#Creating Edge Features
def mask_embedding(embedding: torch.Tensor, missing: torch.Tensor, max_len:int) -> torch.Tensor:
    '''Takes the:
        * pairwise embedding (BXLXLX128), 
        * missing (BXL) binary tensor stored in batch_gt["missing"], 
        * and the max_len (largest sum in missing)
    and creates a masked embedding with shape(B X max_len X max_len X 128) "masked" embedding 
    to match the shape of outputs["single"] and final position. This is the same process seen 
    here: https://github.itap.purdue.edu/kiharalab/af2_e2e/blob/e255e5cdbc5ebc840f9c1b41b5d1285a8c58cf15/model/alphafold_finetune.py#L507
    but for two dimensions.
    '''
    cropped_embedding = torch.zeros_like(embedding, device = embedding.device) # B x L x L x128
    cropped_embedding = cropped_embedding[:,:max_len, :max_len, :] # B x max_len x max_len x128
    for b in range(embedding.size(0)): # Iterate over batch dim (b = 0,1,2,3)
        pad = embedding[b] # L x L X 128
        mask_2d = (missing[b].reshape(-1, 1) * missing[b]) #L X L
        mask_b = pad[mask_2d.bool()].reshape(int(missing[b].sum()), 
                                            int(missing[b].sum()), 
                                            embedding.size(-1)) # 2D mask flattens into missing.sum*missing.sum X feats,
                                                                # need to turn back into missing.sum X missing.sum X Feats

        cropped_embedding[b, :mask_b.size(0),:mask_b.size(1), ...] = mask_b  #Sets this cropped batch in the new output [b, 0:missing.sum,0:missing.sum, 364]
    return cropped_embedding


def block_edge_feats(embedding: torch.Tensor) -> torch.Tensor:
    '''Takes in an embedding tensor (Batch X Length X Length X 128Feats)
     and  returns diagonally blocked  edge features (B*L X B*L X 128Feats)'''

    batches = embedding.shape[0]
    L = embedding.shape[1]
    feats = embedding.shape[3]

    cont_batch = torch.split(embedding, 1) #Split batches into tuples
    cont_batch = [t.reshape(L, L, feats) for t in cont_batch] #Make them all LxL
    
    #torch.block_diag does not work for 3D tensors, need to make manually
    cont_all = torch.zeros(L*batches, L*batches, feats, device = embedding.device) # Create B*L x B*L X F zeros matrix
    for batchnum, batch in enumerate(cont_batch): #For each batch
        start = batchnum * L
        cont_all[start:start+L, start:start+L, :] = batch #Fill in the diagonal with the data from the batch

    return cont_all

def get_edge_attr(blocked_edge_feats: torch.Tensor,
                  edge_index: torch.Tensor) -> torch.Tensor:
    '''Takes the blocked edge features (B*L X B*L X 128Feats)
    and the edge indexes (2 x num_edges) and returns a num_edges X 128Feats 
    tensor of edge attributes. It effecectively pulls out the indicies listed
    in edge_index from blocked_edge_feats'''

    edge_list = edge_index.T.long()
    edge_attr = blocked_edge_feats[edge_list[:,0],edge_list[:,1], :]
    return edge_attr

#Creating Nodes
def batch_nodes(out_single: torch.Tensor, feat_size:int = 384):
    '''Reshapes the single output (BxLx384) 
       to be L*Bx384 (should just stack batches)'''
    return out_single.reshape(-1, feat_size)



def visualize_graph(G, color):
    '''Returns a matplotlib.plt object'''
    plt.figure(figsize=(16,16))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True,
                     node_color=color, cmap="Set2")
    #plt.show()
    return plt

def visualize_embedding(h, color = "red", epoch=None, loss=None):
    '''Copied from some google notebook. Will figure out what it does later'''
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=70, c=color, cmap="Set2")
    plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


def create_graph_figure(in_data):
    '''Returns a matplotlib.fig object of a graph'''
    y = [i for i in range(0, in_data.x.shape[0])] #Color by residue index
    G = to_networkx(in_data, to_undirected=True)
    plt_object = visualize_graph(G, color=y)
    return plt_object


def smart_save_fig(plot, save_dir):
    """Saves figures from a matplotlib plot object (plot) to 
    a directory (save_dir) while avoiding overwriting previous figures"""
    files = os.listdir(save_dir)
    if len(files) == 0:
        new_file = os.path.join(save_dir,"000001.png")
    else:
        indexes = [int(x.replace(".png", "").lstrip("0")) for x in files ]
        last_fig = max(indexes)
        new_index = str(last_fig + 1).rjust(6, "0")
        new_file = os.path.join(save_dir, "{}.png".format(new_index))
    plot.savefig(new_file)
    plot.close()


def smart_save_graph(graph, save_dir):
    """saves torch_geometric graphs (graph) to a directory (save_dir)  
    while avoiding overwriting previous files"""
    files = os.listdir(save_dir)
    if len(files) == 0:
        new_file = os.path.join(save_dir,"1.pytorch")
    else:
        indexes = [int(x.replace(".pytorch", "")) for x in files ]
        last_fig = max(indexes)
        new_file = os.path.join(save_dir, "{}.pytorch".format(str(last_fig + 1)))
    torch.save(graph,new_file)