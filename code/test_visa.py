import os
from model.openllama import OpenLLAMAPEFTModel
import torch
from torchvision import transforms
from sklearn.metrics import roc_auc_score,average_precision_score
from PIL import Image
import numpy as np
import csv
import argparse
from tqdm import tqdm
from metrics import cal_pro_score
# from visualization import visualizer


parser = argparse.ArgumentParser("AnomalyGPT", add_help=True)
parser.add_argument("--few_shot", type=bool, default=True)
parser.add_argument("--k_shot", type=int, default=1)
parser.add_argument("--round", type=int, default=14) # 1-shot:14, 2-shot:57, 4-shot:78

command_args = parser.parse_args()

describles = {
    'candle': 'candle',
    'capsules': 'capsule',
    'cashew': 'cashew',
    'chewinggum': 'chewinggom',
    'fryum': 'fryum',
    'macaroni1': 'macaroni',
    'macaroni2': 'macaroni',
    'pcb1': 'pcb',
    'pcb2': 'pcb',
    'pcb3': 'pcb',
    'pcb4': 'pcb',
    'pipe_fryum': 'pipe fryum'
}


FEW_SHOT = command_args.few_shot

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
    'anomalygpt_ckpt_path': './ckpt/train_mvtec/train_on_mvtec.pt',
    'stage': 2,
    'features_list': [6, 12, 18, 24],
}


model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.cuda()

def predict(
        input,
        image_path,
        normal_img_path,
        r,
):
    prompt_text = input
    pixel_output, cls_score = model.generate({
        'prompt': prompt_text,
        'image_paths': [image_path] if image_path else [],
        'normal_img_paths': normal_img_path if normal_img_path else [],
        'r': r
    })

    return pixel_output, cls_score


root_dir = '/mnt/sda/fenfangtao/VisA'

mask_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

datas_csv_path = '/mnt/sda/fenfangtao/VisA/split_csv/1cls.csv'

CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
               'pcb3', 'pcb4', 'pipe_fryum']

file_paths = {}
normal_img_path = {}

for class_name in CLASS_NAMES:
    file_paths[class_name] = []
    normal_img_path[class_name] = []

with open(datas_csv_path, 'r') as file:
    reader = csv.reader(file)

    for row in reader:
        if row[1] == 'test' and row[0] in CLASS_NAMES:
            file_paths[row[0]].append(os.path.join(root_dir, row[3]))
        if row[0] in CLASS_NAMES and len(normal_img_path[row[0]]) < command_args.round * 4 + command_args.k_shot and \
                row[1] == 'train':
            normal_img_path[row[0]].append(os.path.join(root_dir, row[3]))

if FEW_SHOT:
    for i in CLASS_NAMES:
        normal_img_path[i] = normal_img_path[i][command_args.round * 4:]
        # normal_img_path[i] = random.sample(normal_img_path[i], command_args.k_shot)

r = 0.1
p_auc_list = []
i_auc_list = []
p_pro_list = []
ap_list = []
for c_name in CLASS_NAMES:
    p_pred = []
    p_label = []
    i_pred = []
    i_label = []
    for file_path in tqdm(file_paths[c_name]):
        if FEW_SHOT:
            model.eval()
            with torch.no_grad():
                anomaly_map, score = predict(describles[c_name], file_path, normal_img_path[c_name], r)
        else:
            anomaly_map, score = predict(describles[c_name], file_path, None, 1)

        is_normal = 'Normal' in file_path.split('/')[-2]

        if is_normal:
            img_mask = Image.fromarray(np.zeros((224, 224)), mode='L')
        else:
            mask_path = file_path.replace('Images', 'Masks')
            mask_path = mask_path.replace('.JPG', '.png')
            img_mask = Image.open(mask_path).convert('L')

        img_mask = mask_transform(img_mask)
        threshold = img_mask.max() / 100
        img_mask[img_mask > threshold], img_mask[img_mask <= threshold] = 1, 0
        img_mask = img_mask.squeeze().reshape(224, 224).cpu().numpy()

        anomaly_map = anomaly_map.reshape(224, 224).detach().cpu().numpy()

        # save_path = '/mnt/sda/fenfangtao/AnomalyGPT-main/code/visualize/'
        # visualizer(file_path, anomaly_map, 224, save_path, c_name,'visa')

        p_label.append(img_mask)
        p_pred.append(anomaly_map)

        i_label.append(1 if not is_normal else 0)

        k = 30
        flat_matrix = anomaly_map.ravel()
        top_k_indices = np.argpartition(-flat_matrix, k)[:k]
        top_k_values = flat_matrix[top_k_indices]
        score1 = np.mean(top_k_values)
        i_pred.append(r * score.cpu().numpy() + (1-r) * score1)
        # i_pred.append(anomaly_map.max())

    p_pred = np.array(p_pred)
    p_label = np.array(p_label)

    i_pred = np.array(i_pred)
    i_label = np.array(i_label)

    p_auroc = round(roc_auc_score(p_label.ravel(), p_pred.ravel()) * 100, 2)
    i_auroc = round(roc_auc_score(i_label.ravel(), i_pred.ravel()) * 100, 2)
    p_pro = round(cal_pro_score(p_label, p_pred) * 100, 2)
    ap = round(average_precision_score(i_label, i_pred)* 100, 2)

    p_auc_list.append(p_auroc)
    i_auc_list.append(i_auroc)
    p_pro_list.append(p_pro)
    ap_list.append(ap)

    print(c_name, "i_AUROC:", i_auroc)
    print(c_name, "p_AUROC:", p_auroc)
    print(c_name, "p_pro:", p_pro)
    print(c_name, "ap:", ap)

print("i_AUROC:", torch.tensor(i_auc_list).mean())
print("p_AUROC:", torch.tensor(p_auc_list).mean())
print("p_PRO:", torch.tensor(p_pro_list).mean())
print('ap', torch.tensor(ap_list).mean())
