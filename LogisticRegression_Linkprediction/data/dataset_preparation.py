import pandas as pd
import networkx as nx
from tqdm import tqdm

def retrieve_unconnected(node_list_1, node_list_2, G):  # Retrieve Unconnected Node Pairs – Negative Samples
    print("Create pairs of unconnected nodes...")
    # combine all nodes in a list
    node_list = node_list_1 + node_list_2

    # remove duplicate items from the list
    node_list = list(dict.fromkeys(node_list))

    # build adjacency matrix
    adj_G = nx.to_numpy_array(G, nodelist = node_list)

    # get unconnected node-pairs
    all_unconnected_pairs = []

    # traverse adjacency matrix
    offset = 0
    for i in tqdm(range(adj_G.shape[0])):
        for j in range(offset,adj_G.shape[1]):
            if i != j:
                if nx.shortest_path_length(G, str(i), str(j)) <=2:
                    if adj_G[i,j] == 0:
                        all_unconnected_pairs.append([node_list[i],node_list[j]])
        offset = offset + 1

    node_1_unlinked = [i[0] for i in all_unconnected_pairs]
    node_2_unlinked = [i[1] for i in all_unconnected_pairs]

    data = pd.DataFrame({'node_1':node_1_unlinked, 'node_2':node_2_unlinked})

    # add target variable 'link'
    data['link'] = 0
    return data


def remove_link_connected(fb_df, G):             # Remove Links from Connected Node Pairs – Positive Samples
    print("Remove links from connected node pairs...")
    initial_node_count = len(G.nodes)

    fb_df_temp = fb_df.copy()

    # empty list to store removable links
    omissible_links_index = []

    for i in tqdm(fb_df.index.values):
    
    # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index = i), "node_1", "node_2", create_using=nx.Graph())
        
        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            fb_df_temp = fb_df_temp.drop(index = i)

    return omissible_links_index


def data_for_model_training(fb_df, omissible_links_index, data):
    # create dataframe of removable edges
    fb_df_ghost = fb_df.loc[omissible_links_index]

    # add the target variable 'link'
    fb_df_ghost['link'] = 1

    #data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)
    data = pd.concat([data, fb_df_ghost], ignore_index=True)
    return data, fb_df_ghost

