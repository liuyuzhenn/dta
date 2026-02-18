import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter
from src.models.graph.modules import MPNN


class MultiNet(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

        self.num_features = configs['num_features']
        self.num_layers = configs['num_layers']
        self.iters = configs['iterations']
        self.use_afn = configs.get("AFN", True)
        assert(len(self.num_features) == len(self.num_layers) and len(self.num_features) == len(self.iters))
        self.mlp = configs['mlp']
        self.direciton_invariance_embedding = configs['RDIB']
        self.ac = torch.relu

        self.mpnns = [MPNN(num_layer, num_feat, self.ac, self.mlp) 
                      for num_layer, num_feat in zip(self.num_layers, self.num_features)]
        self.mpnns = nn.ModuleList(self.mpnns)
        
        if self.configs.get('pretrain', '') != '':
            weight = torch.load(configs['pretrain'], map_location='cpu')['model_state_dict']
            self.load_state_dict(weight)
            print('Loading weights from {}'.format(configs['pretrain']))

    def forward(self, data):
        edge_index, edge_attr = data['inds'][0], data['t_rel'][0]
        num_images = data['t_gt'].shape[1]

        # direction invariance expansion
        if self.direciton_invariance_embedding:
            edge_index_inv = edge_index.clone()
            edge_index_inv[0,:] = edge_index[1, :]
            edge_index_inv[1,:] = edge_index[0, :]
            edge_index = torch.concat((edge_index, edge_index_inv), dim=1)
            edge_attr_inv = - edge_attr.clone()
            edge_attr = torch.concat((edge_attr, edge_attr_inv), dim=0)

        if 't_init' in data.keys():
            x = data['t_init'][0]
            if self.configs.get('t_init_normalize', False):
                x = x - torch.mean(x, dim=0, keepdim=True)
                x = x/torch.mean(torch.norm(x, p=2, dim=-1))
        else:
            x = torch.zeros((num_images, 3), device=edge_attr.device)
        
        translations = []
        translations_group = []
        for i in range(len(self.iters)):
            iterations = self.iters[i]
            model = self.mpnns[i]
            group = []
            for it in range(iterations):
                dx = model(x.detach(), edge_attr, edge_index)[0]
                x = x+dx
                if self.use_afn:
                    x = x - torch.mean(x, dim=0, keepdim=True)
                    x = x/torch.mean(torch.norm(x, p=2, dim=-1))
                translations.append(x)
                group.append(x)
            translations_group.append(group)

        mdict = {
            't': x,
            't_iters': translations,
            't_iters_group': translations_group,
        }
        return mdict


if __name__ == "__main__":
    import yaml
    with open("configs/default.yml", "r") as f:
        cfg = yaml.full_load(f)
    a = MultiNet(cfg["model_configs"])
    print(a)
