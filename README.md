# vision.pjt

## 작업 환경

- requiremets.txt 참고

모든 Train과 Evaluation의 경우 데이터를 참고할 경로에 해당 이미지가 있어야 합니다.
- Skin Data의 경우 config/*_skin.yaml의 data_dir가 실제 이미지가 들어간 절대 경로가 포함되어야 하며, csv_path의 csv 파일의 image_name이 data_dir 아래의 image_name과 같아야 합니다. 자세한 내용은 data/skin_filter.csv 참고
- Lung Data의 경우 'image_link'에 해당 이미지의 절대 경로가 들어가야 합니다. 자세한 내용은 data/lung_5_129.csv 참고

### Train

```
python train.py --config {config_path} --output {output_path} --csv-path {csv_path}
```

- 예시
```
##### Skin Data
python train.py --config config/in3_skin.yaml --output output --csv-path skin_filter.csv
python train.py --config config/vit_skin.yaml --output output --csv-path skin_filter.csv
python train.py --config config/pit_skin.yaml --output output --csv-path skin_filter.csv

##### Lung Data
python train.py --config config/in3_lung.yaml --output output --csv-path lung_5_129.csv
python train.py --config config/vit_lung.yaml --output output --csv-path lung_5_129.csv
python train.py --config config/pit_lung.yaml --output output --csv-path lung_5_129.csv

##### Focal Loss
python train.py --config config/in3_skin.yaml --output output --csv-path skin_filter.csv --loss focal

##### Augmentation (default : Augmentation True)
python train.py --config config/in3_skin.yaml --output output --csv-path skin_filter.csv --augment none # Augmentation False
```

### Evaluation
```
python eval.py --config {config_path} --output {output_path} --csv-path {csv_path} --initial_checkpoint {best_model_path}
```

- 예시
```
##### Skin Data
python eval.py --config config/in3_skin.yaml --output eval --csv-path skin_filter.csv --initial-checkpoint best_model_path.tar
python eval.py --config config/vit_skin.yaml --output eval --csv-path skin_filter.csv --initial-checkpoint best_model_path.tar
python eval.py --config config/pit_skin.yaml --output eval --csv-path skin_filter.csv --initial-checkpoint best_model_path.tar

##### Lung Data
python eval.py --config config/in3_lung.yaml --output eval --csv-path lung_5_129.csv --initial-checkpoint best_model_path.tar
python eval.py --config config/vit_lung.yaml --output eval --csv-path lung_5_129.csv --initial-checkpoint best_model_path.tar
python eval.py --config config/pit_lung.yaml --output eval --csv-path lung_5_129.csv --initial-checkpoint best_model_path.tar
```