from torchvision import transforms, datasets
import numpy as np
import os, torch
from sklearn.metrics import accuracy_score
no_transform = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
svhn_train = datasets.OxfordIIITPet('../../../pets/', split='trainval', transform=no_transform, download=True)
svhn_trainloader = torch.utils.data.DataLoader(svhn_train, batch_size=256, shuffle=False)
svhn_test = datasets.OxfordIIITPet('../../../pets/', split='test', transform=no_transform, download=True)
svhn_testloader = torch.utils.data.DataLoader(svhn_test, batch_size=256, shuffle=False)
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet101, ResNet101_Weights
from tqdm import tqdm
models = {
    18 : resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
    50 : resnet101(weights=ResNet101_Weights.IMAGENET1K_V2),
    101 : resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
}
for num, model in models.items():
    model.fc = torch.nn.Identity()
    model.cuda()
    model.eval()
    embs = []
    lbls = []
    with tqdm(total=len(svhn_testloader)) as t:
        for batch in svhn_testloader:
                x, target = batch
                x = x.cuda()
                emb = model(x)
                embs.extend(emb.cpu().detach().numpy())
                lbls.extend(target.numpy())
                t.update()
    embs = np.array(embs)
    lbls = np.array(lbls)
    np.save(f'pets_test_emb{num}.npy', embs)
    if not os.path.exists('pets_test_lbl.npy'):
        np.save(f'pets_test_lbl.npy', lbls)
    embs = []
    lbls = []
    with tqdm(total=len(svhn_trainloader)) as t:
        for batch in svhn_trainloader:
                x, target = batch
                x = x.cuda()
                emb = model(x)
                embs.extend(emb.cpu().detach().numpy())
                lbls.extend(target.numpy())
                t.update()
    embs = np.array(embs)
    lbls = np.array(lbls)
    np.save(f'pets_train_emb{num}.npy', embs)
    if not os.path.exists('pets_train_lbl.npy'):
        np.save(f'pets_train_lbl.npy', lbls)
    model.to('cpu')

import faiss
def get_knn_scor(train_embs, train_target, val_embs, val_target, k=1):
    d = train_embs.shape[1]
    try:
        res = faiss.StandardGpuResources()
        ## Using a flat index
        index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index
        # make it a flat GPU index
        index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        index_flat.add(train_embs) 
    except:
        index_flat = faiss.IndexFlatL2(d) 
        index_flat.add(train_embs) 
    D, I = index_flat.search(val_embs, k=1000)
    returned_labels = train_target[I][:, :k]
    # print(returned_labels[:5, :])
    preds = np.apply_along_axis(get_pred, 1, returned_labels)
    acc = accuracy_score(val_target, preds)
    return acc

def get_pred(v):
    v_u, v_c = np.unique(v, return_counts=True)
    return v_u[v_c.argmax()]