from torch.utils.data import Dataset
import torch
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt
import os

class GPS_Dataset_Wrapper(Dataset):
    def __init__(self,
                 dataset_name,
                 dataset, 
                 sim_matrix,
                 dist_matrix, 
                 cluster_lbls=None, 
                 nn_threshold=-1, 
                 num_nns=1, 
                 num_nns_choice=1, 
                 filter_sim_matrix=True, 
                 subsample_by=1, 
                 clustering_algo=None, 
                 extra_info={},
                 plot_distances=False,
                 save_path='./',
                 no_reloads=None,
                 hop=0) -> None:
        super().__init__()

        self.dataset_name = dataset_name
        self.num_nns = num_nns
        self.num_nns_choice = num_nns_choice
        self.nn_threshold = nn_threshold
        self.dist_matrix = dist_matrix
        self.clusters_as_labels = False
        self.plot_distances = plot_distances
        self.save_path = save_path
        self.no_reloads = no_reloads
        assert num_nns_choice >= num_nns

        self.sim_matrix = sim_matrix
        self.clusters = cluster_lbls
        self.hop = hop
        self.clustering_algo = clustering_algo
        self.extra_info = extra_info
        if self.clusters is not None:
            assert (self.clustering_algo is not None) 
        

                
        self.subsample_by = subsample_by
        if subsample_by > 1:
            self.__subsample_dataset()

        self.dataset = dataset
        self.dataset_type = type(self.dataset).__bases__[0]
        self.labels = self.__get_labels()

        if filter_sim_matrix:
            self._filter_sim_matrix_omit_self()
        else:
            self.sim_matrix = sim_matrix[:, :-1]
            self.dist_matrix = self.dist_matrix[:, :-1]

        self.not_from_cluster_percentage = {'avg': 0, 'var': 0, 'median': 0 }
        self.no_nns = {'avg': 0, 'var': 0, 'median': 0, 'max': 0, 'min': 0}
        self._filter_sim_matrix_by_nnc()

        
        self.relevant_classes = self.get_class_percentage()
        

    def get_class_percentage(self):
        total_lengths = []
        correct_lbls = []

        if self.clusters_as_labels:
            # TODO
            different_clusters = np.unique(self.clusters)
            for c in different_clusters:
                cluster_idx = np.argwhere(self.clusters == c).flatten()
                ...
            # end TODO
        else:
            for idx in range(len(self.sim_matrix)):
                sim_row = self.sim_matrix[idx]
                all_lbls_sim_matrix = self.labels[sim_row]
                # all_lbls_true = self.labels.repeat(self.num_nns_choice).reshape(-1, self.num_nns_choice)
                correct_lbls.append(sum(self.labels[idx] == all_lbls_sim_matrix))
                total_lengths.append(len(sim_row))

        correct_lbls = np.array(correct_lbls, dtype=np.float32)
        total_lengths = np.array(total_lengths, dtype=np.float32)
        avg = correct_lbls.sum() / total_lengths.sum()
        all_percentages = correct_lbls / total_lengths
        median = np.median(all_percentages)
        var = np.var(all_percentages)
        
        return {'avg': avg, 'median': median, 'var': var}

    
    def __subsample_dataset(self):
        # currently only for inat
        if self.dataset_type is not datasets.INaturalist:
            print(f'Subsampling not supported for {self.dataset_type}')
            return
        
        labels = self.__get_labels()
        imgs = np.array(list(list(zip(*self.dataset.index))[1]))
        no_classes = len(np.unique(labels))
        assert no_classes > labels.max()
        new_no_classes = no_classes // self.subsample_by
        new_imgs = imgs[labels <= new_no_classes]
        new_labels = labels[labels <= new_no_classes]
        new_index = list(zip(new_labels, new_imgs))
        self.dataset.index = new_index
        return
  

    def __get_labels(self):
        if self.dataset_type is datasets.CIFAR10 or \
            self.dataset_type is datasets.CIFAR100:
            return np.array(self.dataset.targets)
        elif self.dataset_type is datasets.SVHN:
            return np.array(self.dataset.labels)
        elif self.dataset_type is datasets.INaturalist:
            return np.array(list(list(zip(*self.dataset.index))[0]))
        elif self.dataset_type is datasets.FGVCAircraft:
            return np.array(self.dataset._labels)
        elif self.dataset_type is datasets.ImageFolder:
            return np.array(list(list(zip(*self.dataset.samples))[1]))
        else:
            return self.dataset.labels


    def __getitem__(self, index):
        if self.clusters_as_labels:
            # TODO
            img_cluster = self.clusters[index]
            candidates = np.argwhere(self.clusters == img_cluster).flatten()
            # end TODO
        else:
            candidates = self.sim_matrix[index]

        sim_index_idxes = np.random.randint(0, len(candidates), self.num_nns)
        sim_index = candidates[sim_index_idxes]
        all_idxs = []
        all_xs = []
        all_ys = []
        idx1, x1, y1 = self.dataset.__getitem__(index)
        if len(x1) != 1:
            print('Warning, More than 1 augmentations found, using first one')
        x1 = x1[0]
        all_idxs.append(idx1)
        all_xs.append(x1)
        all_ys.append(y1)

        for i in sim_index:
            idx2, x2, y2 = self.dataset.__getitem__(i)
            x2 = x2[0]
            all_idxs.append(idx2)
            all_xs.append(x2)
            all_ys.append(y2)

        return all_idxs, all_xs, all_ys


    def _filter_sim_matrix_omit_self(self):
        """
        remove datapoint itself from its nearest neighbors (default should be False)
        """
        new_sim_idices = []
        new_dist_idices = []
        for idx, row in enumerate(self.sim_matrix):
            if idx in row:
                new_row = np.concatenate([row[:np.argwhere(row == idx)[0][0]], row[np.argwhere(row == idx)[0][0] + 1:]])
                new_dist_row = np.concatenate([self.dist_matrix[idx][:np.argwhere(row == idx)[0][0]], self.dist_matrix[idx][np.argwhere(row == idx)[0][0] + 1:]])
                
            else:
                new_row = row[:-1]
                new_dist_row = self.dist_matrix[idx][:-1]

            new_sim_idices.append(new_row)
            new_dist_idices.append(new_dist_row)
        
        new_sim_matrix = np.stack(new_sim_idices, axis=0)
        new_dist_matrix = np.stack(new_dist_idices, axis=0)

        assert new_sim_matrix.shape[0] == self.sim_matrix.shape[0]
        assert new_sim_matrix.shape[1] == (self.sim_matrix.shape[1] - 1)
        
        assert new_dist_matrix.shape[0] == self.dist_matrix.shape[0]
        assert new_dist_matrix.shape[1] == (self.dist_matrix.shape[1] - 1)
        
        self.sim_matrix = new_sim_matrix
        self.dist_matrix = new_dist_matrix
        return
    
    def _update_nns_stats(self):
        nns = []
        for idx, row in enumerate(self.sim_matrix):
            nns.append(len(row))

        no_nns = np.array(nns)

        self.no_nns['avg'] = no_nns.mean()
        self.no_nns['median'] = np.median(no_nns)
        self.no_nns['var'] = np.var(no_nns)
        self.no_nns['max'] = np.max(no_nns)
        self.no_nns['min'] = np.min(no_nns)

        if self.plot_distances:
            self._plot_pos_neg_hist()

        return

    def _plot_pos_neg_hist(self, bins=300):
        print(f'Plotting pos neg histograms on reload {self.no_reloads}...')
        plt.clf()
        correct_lbls = []
        pos_dist_matrix = []
        neg_dist_matrix = []
        for idx, sim_row in enumerate(self.sim_matrix):
            correct_lbls = (self.labels[idx] == self.labels[sim_row])
            pos_dist_matrix.extend(self.dist_matrix[idx][correct_lbls])
            neg_dist_matrix.extend(self.dist_matrix[idx][np.logical_not(correct_lbls)])

        plt.hist(neg_dist_matrix, bins=bins, color='r', alpha=0.3)
        plt.hist(pos_dist_matrix, bins=bins, color='g', alpha=0.3)
        if self.no_reloads is not None:
            title = f'{self.dataset_name}_reloads{self.no_reloads}'
        else:
            title = self.dataset_name
        plt.title(f'{title}')
        plt.savefig(os.path.join(self.save_path, f'{title}.pdf'))

        plt.clf()
        plt.hist(neg_dist_matrix, bins=bins, color='r', alpha=0.3)
        plt.hist(pos_dist_matrix, bins=bins, color='g', alpha=0.3)
        plt.yscale('log')
        plt.title(f'{title} Log Scale')
        plt.savefig(os.path.join(self.save_path, f'{title}_log.pdf'))

    def _filter_sim_matrix_by_nnc(self):
        not_from_cluster = []

        if self.nn_threshold > 0 and (not self.clustering_algo.startswith('louvain')):
            new_dist_list = []
            new_sim_list = []
            for idx, row in enumerate(self.sim_matrix):
                
                dist_row = self.dist_matrix[idx]

                new_row = row[dist_row <= self.nn_threshold]
                new_dist_row = dist_row[dist_row <= self.nn_threshold]
                if len(new_row) == 0:
                    new_row = np.array([idx])
                    new_dist_row = np.array([0.0])
                    
                new_sim_list.append(new_row)
                new_dist_list.append(new_dist_row)

            # new_sim_matrix = np.stack(new_sim_idices, axis=0)
            # new_dist_matrix = np.stack(new_dist_list, axis=0)
            assert len(new_sim_list) == len(self.sim_matrix)
            assert len(new_dist_list) == len(self.dist_matrix)
             

            self.sim_matrix = new_sim_list
            self.dist_matrix = new_dist_list

        if self.clusters is not None and not self.clustering_algo.startswith('louvain'):
            new_dist_list = []
            new_sim_list = []
            for idx, row in enumerate(self.sim_matrix):
                row_clusters = self.clusters[row].flatten()
                
                idx_from_same_cluster = row[row_clusters == row_clusters[0]]
                new_row = idx_from_same_cluster[:self.num_nns_choice]

                idx_from_same_cluster_dist = self.dist_matrix[idx][row_clusters == row_clusters[0]]
                if self.clustering_algo.startswith('louvain'):
                    new_dist_row = idx_from_same_cluster_dist
                else:
                    new_dist_row = idx_from_same_cluster_dist[:self.num_nns_choice]

                new_sim_list.append(new_row)
                new_dist_list.append(new_dist_row)
                
                not_from_cluster.append(len(set(row[:self.num_nns_choice]) - set(new_row)))
            
            assert len(new_sim_list) == len(self.sim_matrix)

            assert len(new_dist_list) == len(self.dist_matrix)

            self.sim_matrix = new_sim_list
            self.dist_matrix = new_dist_list

            not_from_cluster = np.array(not_from_cluster) / self.num_nns_choice 

            self.not_from_cluster_percentage['avg'] = not_from_cluster.mean()
            self.not_from_cluster_percentage['median'] = np.median(not_from_cluster)
            self.not_from_cluster_percentage['var'] = np.var(not_from_cluster)
        
        if self.clusters is not None and self.clustering_algo.startswith('louvain'):
            new_dist_list = []
            new_sim_list = []
            all_image_row = np.array([i for i in range(len(self.clusters))])
            self.clusters_as_labels = True
            for idx, row in enumerate(self.sim_matrix):
                
                new_row = all_image_row[self.clusters[idx] == self.clusters]

                new_sim_list.append(new_row)
                
                not_from_cluster.append(len(set(row[:self.num_nns_choice]) - set(new_row)))
            
            
            self.sim_matrix = new_sim_list

            not_from_cluster = np.array(not_from_cluster) / self.num_nns_choice 

            self.not_from_cluster_percentage['avg'] = not_from_cluster.mean()
            self.not_from_cluster_percentage['median'] = np.median(not_from_cluster)
            self.not_from_cluster_percentage['var'] = np.var(not_from_cluster)
        
        if self.hop > 0:
            adj_matrix = np.zeros((len(self.sim_matrix), len(self.sim_matrix)), dtype=np.float64)
            x_idx = np.array([i for i in range(len(self.sim_matrix))])
            x_idx = x_idx.repeat(self.num_nns_choice)
            adj_matrix[x_idx, self.sim_matrix[:, :self.num_nns_choice].flatten()] = 1
            hop_matrix = np.identity(len(self.sim_matrix), dtype=np.float64)
            adj_matrix = torch.tensor(adj_matrix).to_sparse()
            hop_matrix = torch.tensor(hop_matrix).to_sparse()
            hop_matrix = torch.sparse.mm(adj_matrix, hop_matrix)
            for _ in range(self.hop):
                hop_matrix = torch.sparse.mm(adj_matrix, hop_matrix)
            

            hop_matrix = hop_matrix.to_dense().numpy()
            new_sim_list = []
            new_dist_list = []

            for h in hop_matrix:
                new_sim_list.append(np.argwhere(h != 0).flatten())
                new_dist_list.append(np.ones(new_sim_list[-1].shape).flatten())

            self.sim_matrix = new_sim_list
            self.dist_matrix = new_dist_list

        if (self.clusters is None) and (self.nn_threshold < 0) and (self.hop == 0):
            new_sim_list = []
            new_dist_list = []
            for idx in range(len(self.sim_matrix)):
                new_sim_list.append(self.sim_matrix[idx][:self.num_nns_choice])
                new_dist_list.append(self.dist_matrix[idx][:self.num_nns_choice])
  
            self.sim_matrix = new_sim_list
            self.dist_matrix = new_dist_list

        self._update_nns_stats()
        return
    



    def __len__(self):
        return len(self.dataset)