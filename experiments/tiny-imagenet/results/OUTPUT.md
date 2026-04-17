$ /home/dgxuser15/.venv/bin/python /home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/scripts/eda.py
Train: 100,000 images / 200 classes (min=500, max=500)
Val: 10,000 images | Test: 10,000 images (unlabelled)
Corruption check (2,000-image sample): 0 corrupt ✓
Computing RGB statistics (5 000-image sample) …
RGB Mean : R=0.4846  G=0.4510  B=0.4010
RGB Std  : R=0.2767  G=0.2686  B=0.2817
✓  All outputs saved to /home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/results/


*************************************************************************************************************************************************************************
*************************************************************************************************************************************************************************


$ /home/dgxuser15/.venv/bin/python /home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/scripts/data-processing.py
Device : cuda
GPU    : Tesla V100-SXM2-32GB
VRAM   : 31.7 GB
TF32   : matmul=True  cudnn=True
Paths configured.
Loaded mean/std from dataset_stats.json
  mean = (0.484622985124588, 0.4509679973125458, 0.4010140001773834)
  std  = (0.2767139971256256, 0.26859501004219055, 0.28167200088500977)
  #classes = 200
Augmentation strategy:
  TRAIN : RandomCrop(64,pad=8) | HFlip(0.5) | ColorJitter | RandomGrayscale | RandomErasing(0.2) | Normalize
  VAL   : CenterCrop(56) → Resize(64) | Normalize
Train samples : 100,000
Val   samples : 10,000
✓  Manifest saved  → /raid/home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/processed/data_manifest.pkl
✓  Summary saved   → /raid/home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/processed/manifest_summary.json
{
  "train_samples": 100000,
  "val_samples": 10000,
  "num_classes": 200,
  "mean": [
    0.484622985124588,
    0.4509679973125458,
    0.4010140001773834
  ],
  "std": [
    0.2767139971256256,
    0.26859501004219055,
    0.28167200088500977
  ]
}
DataLoader config:
  num_workers = 8 | pin_memory = True
  batch_size  = 256  | drop_last (train) = True
  train batches : 390 | val batches : 40
Generating augmentation preview …
Benchmarking DataLoader (batch=256, 20 batches) …
  Throughput : 9,164 imgs/sec
  Workers    : 8
  Elapsed    : 0.56s for 5,120 images
Saved → dataloader_benchmark.json


*************************************************************************************************************************************************************************
*************************************************************************************************************************************************************************


$ /home/dgxuser15/.venv/bin/python /home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/scripts/model-training.py
Device : cuda
GPU    : Tesla V100-SXM2-32GB
VRAM   : 31.7 GB
TF32   : matmul=True  cudnn=True
Classes : 200  |  mean=(0.484622985124588, 0.4509679973125458, 0.4010140001773834)  |  std=(0.2767139971256256, 0.26859501004219055, 0.28167200088500977)
Config: {
  "batch_size": 256,
  "epochs": 50,
  "lr": 0.003,
  "weight_decay": 0.0005,
  "label_smoothing": 0.1,
  "mixup_alpha": 0.3,
  "patience": 15,
  "num_workers": 8,
  "pin_memory": true,
  "grad_clip": 5.0
}
DRY_RUN = False
Train batches : 390 | Val batches : 40
build_model() ready | channels_last=True | torch.compile=True

============================================================
  Training : MOBILENETV2  |  Device: cuda
============================================================
  [compile] torch.compile() applied to mobilenetv2
  Parameters : 2.48 M

Epoch [01/50]  TrainLoss=5.1059  ValLoss=4.9494  Acc@1=3.08%  Acc@5=12.69%  lr=0.00300  t=142.6s
Epoch [02/50]  TrainLoss=4.7788  ValLoss=4.6016  Acc@1=6.26%  Acc@5=22.15%  lr=0.00299  t=28.5s
Epoch [03/50]  TrainLoss=4.6095  ValLoss=4.3722  Acc@1=9.96%  Acc@5=29.67%  lr=0.00297  t=27.4s
Epoch [04/50]  TrainLoss=4.4270  ValLoss=4.1901  Acc@1=13.23%  Acc@5=35.46%  lr=0.00295  t=27.2s
Epoch [05/50]  TrainLoss=4.2797  ValLoss=3.9738  Acc@1=17.55%  Acc@5=42.25%  lr=0.00293  t=27.5s
Epoch [06/50]  TrainLoss=4.1758  ValLoss=3.8178  Acc@1=21.20%  Acc@5=46.72%  lr=0.00289  t=26.8s
Epoch [07/50]  TrainLoss=4.0555  ValLoss=3.6646  Acc@1=23.90%  Acc@5=51.60%  lr=0.00286  t=27.4s
Epoch [08/50]  TrainLoss=3.9736  ValLoss=3.6202  Acc@1=25.54%  Acc@5=53.01%  lr=0.00281  t=26.8s
Epoch [09/50]  TrainLoss=3.9173  ValLoss=3.4693  Acc@1=28.61%  Acc@5=56.58%  lr=0.00277  t=26.5s
Epoch [10/50]  TrainLoss=3.8104  ValLoss=3.5103  Acc@1=28.14%  Acc@5=56.74%  lr=0.00271  t=26.3s
Epoch [11/50]  TrainLoss=3.8103  ValLoss=3.3419  Acc@1=32.03%  Acc@5=60.77%  lr=0.00266  t=26.9s
Epoch [12/50]  TrainLoss=3.7542  ValLoss=3.3403  Acc@1=32.19%  Acc@5=61.13%  lr=0.00259  t=27.2s
Epoch [13/50]  TrainLoss=3.7047  ValLoss=3.2692  Acc@1=33.74%  Acc@5=62.32%  lr=0.00253  t=27.3s
Epoch [14/50]  TrainLoss=3.5974  ValLoss=3.2020  Acc@1=35.39%  Acc@5=63.91%  lr=0.00246  t=27.5s
Epoch [15/50]  TrainLoss=3.6090  ValLoss=3.1702  Acc@1=36.54%  Acc@5=64.43%  lr=0.00238  t=28.1s
Epoch [16/50]  TrainLoss=3.5389  ValLoss=3.1542  Acc@1=36.99%  Acc@5=65.29%  lr=0.00230  t=27.2s
Epoch [17/50]  TrainLoss=3.5327  ValLoss=3.0987  Acc@1=38.16%  Acc@5=66.41%  lr=0.00222  t=28.9s
Epoch [18/50]  TrainLoss=3.4994  ValLoss=3.0852  Acc@1=38.51%  Acc@5=66.65%  lr=0.00214  t=27.7s
Epoch [19/50]  TrainLoss=3.4581  ValLoss=3.0468  Acc@1=39.68%  Acc@5=68.38%  lr=0.00205  t=31.5s
Epoch [20/50]  TrainLoss=3.4669  ValLoss=3.0359  Acc@1=39.98%  Acc@5=68.16%  lr=0.00196  t=34.4s
Epoch [21/50]  TrainLoss=3.4511  ValLoss=3.0362  Acc@1=40.47%  Acc@5=68.69%  lr=0.00187  t=34.3s
Epoch [22/50]  TrainLoss=3.3739  ValLoss=2.9434  Acc@1=41.96%  Acc@5=70.22%  lr=0.00178  t=34.3s
Epoch [23/50]  TrainLoss=3.3655  ValLoss=2.9498  Acc@1=41.80%  Acc@5=69.98%  lr=0.00169  t=31.2s
Epoch [24/50]  TrainLoss=3.2954  ValLoss=2.9187  Acc@1=42.92%  Acc@5=70.87%  lr=0.00159  t=27.3s
Epoch [25/50]  TrainLoss=3.2803  ValLoss=2.8473  Acc@1=44.12%  Acc@5=72.02%  lr=0.00150  t=26.6s
Epoch [26/50]  TrainLoss=3.2402  ValLoss=2.8653  Acc@1=44.38%  Acc@5=72.11%  lr=0.00141  t=26.6s
Epoch [27/50]  TrainLoss=3.2717  ValLoss=2.8660  Acc@1=44.77%  Acc@5=72.40%  lr=0.00131  t=26.6s
Epoch [28/50]  TrainLoss=3.1761  ValLoss=2.8101  Acc@1=45.50%  Acc@5=72.89%  lr=0.00122  t=27.0s
Epoch [29/50]  TrainLoss=3.1973  ValLoss=2.8160  Acc@1=45.63%  Acc@5=73.71%  lr=0.00113  t=27.3s
Epoch [30/50]  TrainLoss=3.2474  ValLoss=2.7850  Acc@1=46.01%  Acc@5=73.52%  lr=0.00104  t=29.1s
Epoch [31/50]  TrainLoss=3.1621  ValLoss=2.7876  Acc@1=46.67%  Acc@5=74.06%  lr=0.00095  t=29.4s
Epoch [32/50]  TrainLoss=3.1042  ValLoss=2.7751  Acc@1=46.67%  Acc@5=73.92%  lr=0.00086  t=29.1s
Epoch [33/50]  TrainLoss=3.0954  ValLoss=2.7353  Acc@1=47.42%  Acc@5=74.79%  lr=0.00078  t=28.1s
Epoch [34/50]  TrainLoss=3.1257  ValLoss=2.7794  Acc@1=47.52%  Acc@5=74.76%  lr=0.00070  t=27.1s
Epoch [35/50]  TrainLoss=3.1103  ValLoss=2.7733  Acc@1=46.91%  Acc@5=74.35%  lr=0.00062  t=27.3s
Epoch [36/50]  TrainLoss=3.0436  ValLoss=2.6982  Acc@1=48.42%  Acc@5=75.67%  lr=0.00054  t=34.4s
Epoch [37/50]  TrainLoss=2.9492  ValLoss=2.6833  Acc@1=48.86%  Acc@5=75.53%  lr=0.00047  t=34.4s
Epoch [38/50]  TrainLoss=3.0410  ValLoss=2.6948  Acc@1=48.64%  Acc@5=75.62%  lr=0.00041  t=34.5s
Epoch [39/50]  TrainLoss=3.0180  ValLoss=2.7050  Acc@1=48.85%  Acc@5=75.14%  lr=0.00034  t=33.6s
Epoch [40/50]  TrainLoss=3.0320  ValLoss=2.7048  Acc@1=48.88%  Acc@5=75.33%  lr=0.00029  t=26.9s
Epoch [41/50]  TrainLoss=3.0479  ValLoss=2.6845  Acc@1=48.45%  Acc@5=75.95%  lr=0.00023  t=26.3s
Epoch [42/50]  TrainLoss=3.0169  ValLoss=2.6835  Acc@1=49.31%  Acc@5=75.69%  lr=0.00019  t=26.1s
Epoch [43/50]  TrainLoss=2.9582  ValLoss=2.6980  Acc@1=49.21%  Acc@5=75.55%  lr=0.00014  t=26.2s
Epoch [44/50]  TrainLoss=3.0262  ValLoss=2.6792  Acc@1=49.31%  Acc@5=75.93%  lr=0.00011  t=26.5s
Epoch [45/50]  TrainLoss=2.9802  ValLoss=2.6730  Acc@1=49.46%  Acc@5=75.93%  lr=0.00007  t=26.7s
Epoch [46/50]  TrainLoss=2.9464  ValLoss=2.6773  Acc@1=49.56%  Acc@5=76.00%  lr=0.00005  t=26.3s
Epoch [47/50]  TrainLoss=2.8850  ValLoss=2.6425  Acc@1=49.59%  Acc@5=76.27%  lr=0.00003  t=26.6s
Epoch [48/50]  TrainLoss=2.9531  ValLoss=2.6566  Acc@1=50.02%  Acc@5=76.06%  lr=0.00001  t=26.8s
Epoch [49/50]  TrainLoss=2.9753  ValLoss=2.7038  Acc@1=48.86%  Acc@5=75.56%  lr=0.00000  t=26.8s
Epoch [50/50]  TrainLoss=3.0438  ValLoss=2.6502  Acc@1=50.20%  Acc@5=75.64%  lr=0.00000  t=26.8s

✓  Metrics → results/baseline_mobilenetv2_metrics.json

============================================================
  Training : SHUFFLENETV2  |  Device: cuda
============================================================
  [compile] torch.compile() applied to shufflenetv2
  Parameters : 1.46 M

Epoch [01/50]  TrainLoss=5.1230  ValLoss=4.7727  Acc@1=5.03%  Acc@5=17.10%  lr=0.00300  t=169.0s
Epoch [02/50]  TrainLoss=4.6986  ValLoss=4.3585  Acc@1=10.83%  Acc@5=30.67%  lr=0.00299  t=26.7s
Epoch [03/50]  TrainLoss=4.4264  ValLoss=4.0722  Acc@1=16.20%  Acc@5=40.28%  lr=0.00297  t=26.5s
Epoch [04/50]  TrainLoss=4.2333  ValLoss=3.9135  Acc@1=19.65%  Acc@5=45.38%  lr=0.00295  t=26.4s
Epoch [05/50]  TrainLoss=4.0782  ValLoss=3.7763  Acc@1=23.23%  Acc@5=49.37%  lr=0.00293  t=26.3s
Epoch [06/50]  TrainLoss=3.9270  ValLoss=3.5569  Acc@1=27.04%  Acc@5=54.46%  lr=0.00289  t=26.1s
Epoch [07/50]  TrainLoss=3.9091  ValLoss=3.5205  Acc@1=28.77%  Acc@5=55.97%  lr=0.00286  t=25.9s
Epoch [08/50]  TrainLoss=3.7951  ValLoss=3.4261  Acc@1=30.13%  Acc@5=58.62%  lr=0.00281  t=26.1s
Epoch [09/50]  TrainLoss=3.7138  ValLoss=3.2984  Acc@1=33.62%  Acc@5=61.81%  lr=0.00277  t=26.0s
Epoch [10/50]  TrainLoss=3.6377  ValLoss=3.2490  Acc@1=35.41%  Acc@5=62.72%  lr=0.00271  t=27.1s
Epoch [11/50]  TrainLoss=3.5676  ValLoss=3.1837  Acc@1=36.58%  Acc@5=64.29%  lr=0.00266  t=26.3s
Epoch [12/50]  TrainLoss=3.5125  ValLoss=3.1481  Acc@1=37.63%  Acc@5=65.05%  lr=0.00259  t=26.3s
Epoch [13/50]  TrainLoss=3.4881  ValLoss=3.1325  Acc@1=38.22%  Acc@5=65.49%  lr=0.00253  t=26.3s
Epoch [14/50]  TrainLoss=3.4130  ValLoss=3.0934  Acc@1=39.26%  Acc@5=66.85%  lr=0.00246  t=26.6s
Epoch [15/50]  TrainLoss=3.3803  ValLoss=3.0237  Acc@1=41.10%  Acc@5=68.55%  lr=0.00238  t=26.3s
Epoch [16/50]  TrainLoss=3.2635  ValLoss=3.0276  Acc@1=40.51%  Acc@5=68.40%  lr=0.00230  t=26.4s
Epoch [17/50]  TrainLoss=3.2382  ValLoss=2.9415  Acc@1=42.58%  Acc@5=69.60%  lr=0.00222  t=26.5s
Epoch [18/50]  TrainLoss=3.2344  ValLoss=3.0126  Acc@1=41.58%  Acc@5=69.07%  lr=0.00214  t=26.6s
Epoch [19/50]  TrainLoss=3.2035  ValLoss=2.9234  Acc@1=43.89%  Acc@5=70.44%  lr=0.00205  t=26.8s
Epoch [20/50]  TrainLoss=3.1432  ValLoss=2.9949  Acc@1=42.06%  Acc@5=69.14%  lr=0.00196  t=27.1s
Epoch [21/50]  TrainLoss=3.1703  ValLoss=2.8982  Acc@1=44.49%  Acc@5=71.67%  lr=0.00187  t=26.7s
Epoch [22/50]  TrainLoss=3.1063  ValLoss=2.9157  Acc@1=44.60%  Acc@5=70.62%  lr=0.00178  t=27.9s
Epoch [23/50]  TrainLoss=3.0691  ValLoss=2.8320  Acc@1=46.36%  Acc@5=72.63%  lr=0.00169  t=28.4s
Epoch [24/50]  TrainLoss=3.0137  ValLoss=2.8519  Acc@1=45.53%  Acc@5=72.02%  lr=0.00159  t=29.0s
Epoch [25/50]  TrainLoss=2.9911  ValLoss=2.8773  Acc@1=45.34%  Acc@5=71.60%  lr=0.00150  t=27.2s
Epoch [26/50]  TrainLoss=2.9649  ValLoss=2.8153  Acc@1=46.92%  Acc@5=72.47%  lr=0.00141  t=27.4s
Epoch [27/50]  TrainLoss=2.9415  ValLoss=2.8326  Acc@1=46.89%  Acc@5=72.71%  lr=0.00131  t=26.7s
Epoch [28/50]  TrainLoss=2.9214  ValLoss=2.7884  Acc@1=47.50%  Acc@5=73.58%  lr=0.00122  t=26.8s
Epoch [29/50]  TrainLoss=2.8808  ValLoss=2.7822  Acc@1=47.93%  Acc@5=73.29%  lr=0.00113  t=27.2s
Epoch [30/50]  TrainLoss=2.9000  ValLoss=2.8078  Acc@1=47.55%  Acc@5=72.86%  lr=0.00104  t=27.0s
Epoch [31/50]  TrainLoss=2.8783  ValLoss=2.7860  Acc@1=47.62%  Acc@5=73.72%  lr=0.00095  t=27.5s
Epoch [32/50]  TrainLoss=2.8480  ValLoss=2.8007  Acc@1=47.81%  Acc@5=73.02%  lr=0.00086  t=26.6s
Epoch [33/50]  TrainLoss=2.8115  ValLoss=2.7646  Acc@1=48.72%  Acc@5=73.74%  lr=0.00078  t=26.7s
Epoch [34/50]  TrainLoss=2.9072  ValLoss=2.8012  Acc@1=47.89%  Acc@5=73.16%  lr=0.00070  t=27.4s
Epoch [35/50]  TrainLoss=2.7677  ValLoss=2.7653  Acc@1=48.33%  Acc@5=74.02%  lr=0.00062  t=28.0s
Epoch [36/50]  TrainLoss=2.6890  ValLoss=2.7582  Acc@1=48.77%  Acc@5=74.08%  lr=0.00054  t=28.4s
Epoch [37/50]  TrainLoss=2.7589  ValLoss=2.7374  Acc@1=48.72%  Acc@5=74.34%  lr=0.00047  t=27.8s
Epoch [38/50]  TrainLoss=2.6958  ValLoss=2.7402  Acc@1=49.08%  Acc@5=74.27%  lr=0.00041  t=27.9s
Epoch [39/50]  TrainLoss=2.7595  ValLoss=2.7487  Acc@1=48.80%  Acc@5=74.32%  lr=0.00034  t=28.7s
Epoch [40/50]  TrainLoss=2.7387  ValLoss=2.7558  Acc@1=49.04%  Acc@5=74.23%  lr=0.00029  t=26.7s
Epoch [41/50]  TrainLoss=2.7212  ValLoss=2.7385  Acc@1=49.74%  Acc@5=74.47%  lr=0.00023  t=26.2s
Epoch [42/50]  TrainLoss=2.7293  ValLoss=2.7253  Acc@1=49.30%  Acc@5=74.88%  lr=0.00019  t=26.4s
Epoch [43/50]  TrainLoss=2.7121  ValLoss=2.7036  Acc@1=49.99%  Acc@5=75.01%  lr=0.00014  t=27.9s
Epoch [44/50]  TrainLoss=2.7290  ValLoss=2.7125  Acc@1=49.99%  Acc@5=75.15%  lr=0.00011  t=27.9s
Epoch [45/50]  TrainLoss=2.6552  ValLoss=2.7154  Acc@1=50.16%  Acc@5=74.93%  lr=0.00007  t=27.8s
Epoch [46/50]  TrainLoss=2.5409  ValLoss=2.7112  Acc@1=50.01%  Acc@5=75.28%  lr=0.00005  t=27.8s
Epoch [47/50]  TrainLoss=2.6410  ValLoss=2.7066  Acc@1=49.93%  Acc@5=74.97%  lr=0.00003  t=27.6s
Epoch [48/50]  TrainLoss=2.6268  ValLoss=2.7298  Acc@1=49.81%  Acc@5=74.76%  lr=0.00001  t=27.4s
Epoch [49/50]  TrainLoss=2.5856  ValLoss=2.6878  Acc@1=50.63%  Acc@5=75.29%  lr=0.00000  t=27.2s
Epoch [50/50]  TrainLoss=2.5739  ValLoss=2.7247  Acc@1=49.19%  Acc@5=74.52%  lr=0.00000  t=26.4s

✓  Metrics → results/baseline_shufflenetv2_metrics.json

============================================================
  Training : EFFICIENTNET_B0  |  Device: cuda
============================================================
  [compile] torch.compile() applied to efficientnet_b0
  Parameters : 4.26 M

Epoch [01/50]  TrainLoss=5.1050  ValLoss=4.7760  Acc@1=4.42%  Acc@5=16.99%  lr=0.00300  t=240.9s
Epoch [02/50]  TrainLoss=4.7416  ValLoss=4.5767  Acc@1=7.71%  Acc@5=24.02%  lr=0.00299  t=26.5s
Epoch [03/50]  TrainLoss=4.4761  ValLoss=4.1030  Acc@1=14.97%  Acc@5=38.51%  lr=0.00297  t=27.0s
Epoch [04/50]  TrainLoss=4.3458  ValLoss=3.9203  Acc@1=19.34%  Acc@5=45.65%  lr=0.00295  t=26.7s
Epoch [05/50]  TrainLoss=4.2124  ValLoss=3.8225  Acc@1=21.70%  Acc@5=47.85%  lr=0.00293  t=26.7s
Epoch [06/50]  TrainLoss=4.0849  ValLoss=3.7200  Acc@1=24.05%  Acc@5=51.74%  lr=0.00289  t=27.6s
Epoch [07/50]  TrainLoss=3.9689  ValLoss=3.6422  Acc@1=26.45%  Acc@5=53.65%  lr=0.00286  t=28.4s
Epoch [08/50]  TrainLoss=3.9388  ValLoss=3.4821  Acc@1=29.77%  Acc@5=58.16%  lr=0.00281  t=26.9s
Epoch [09/50]  TrainLoss=3.8118  ValLoss=3.4154  Acc@1=30.36%  Acc@5=58.84%  lr=0.00277  t=26.5s
Epoch [10/50]  TrainLoss=3.6896  ValLoss=3.3158  Acc@1=33.30%  Acc@5=61.79%  lr=0.00271  t=26.3s
Epoch [11/50]  TrainLoss=3.6187  ValLoss=3.1906  Acc@1=35.55%  Acc@5=64.89%  lr=0.00266  t=26.3s
Epoch [12/50]  TrainLoss=3.6412  ValLoss=3.1844  Acc@1=36.30%  Acc@5=64.90%  lr=0.00259  t=26.2s
Epoch [13/50]  TrainLoss=3.5962  ValLoss=3.1312  Acc@1=37.88%  Acc@5=66.17%  lr=0.00253  t=26.3s
Epoch [14/50]  TrainLoss=3.5100  ValLoss=3.1571  Acc@1=37.36%  Acc@5=65.14%  lr=0.00246  t=26.5s
Epoch [15/50]  TrainLoss=3.4875  ValLoss=3.0941  Acc@1=39.00%  Acc@5=66.82%  lr=0.00238  t=26.6s
Epoch [16/50]  TrainLoss=3.4502  ValLoss=3.0158  Acc@1=40.38%  Acc@5=68.31%  lr=0.00230  t=26.7s
Epoch [17/50]  TrainLoss=3.3751  ValLoss=2.9425  Acc@1=42.76%  Acc@5=70.11%  lr=0.00222  t=26.5s
Epoch [18/50]  TrainLoss=3.3560  ValLoss=2.9435  Acc@1=42.89%  Acc@5=70.24%  lr=0.00214  t=26.6s
Epoch [19/50]  TrainLoss=3.3188  ValLoss=2.8936  Acc@1=44.13%  Acc@5=71.04%  lr=0.00205  t=27.0s
Epoch [20/50]  TrainLoss=3.3076  ValLoss=2.8807  Acc@1=44.40%  Acc@5=71.45%  lr=0.00196  t=26.8s
Epoch [21/50]  TrainLoss=3.2467  ValLoss=2.9001  Acc@1=44.25%  Acc@5=71.48%  lr=0.00187  t=27.0s
Epoch [22/50]  TrainLoss=3.1544  ValLoss=2.8658  Acc@1=44.21%  Acc@5=71.51%  lr=0.00178  t=27.1s
Epoch [23/50]  TrainLoss=3.2328  ValLoss=2.7793  Acc@1=46.58%  Acc@5=73.09%  lr=0.00169  t=26.7s
Epoch [24/50]  TrainLoss=3.0904  ValLoss=2.7589  Acc@1=47.70%  Acc@5=73.77%  lr=0.00159  t=26.6s
Epoch [25/50]  TrainLoss=3.0803  ValLoss=2.7465  Acc@1=48.54%  Acc@5=73.94%  lr=0.00150  t=28.5s
Epoch [26/50]  TrainLoss=3.0655  ValLoss=2.7573  Acc@1=47.86%  Acc@5=74.17%  lr=0.00141  t=26.9s
Epoch [27/50]  TrainLoss=2.9709  ValLoss=2.7845  Acc@1=47.70%  Acc@5=73.76%  lr=0.00131  t=26.6s
Epoch [28/50]  TrainLoss=3.0160  ValLoss=2.6910  Acc@1=49.11%  Acc@5=75.27%  lr=0.00122  t=28.5s
Epoch [29/50]  TrainLoss=2.9689  ValLoss=2.7265  Acc@1=48.25%  Acc@5=75.04%  lr=0.00113  t=26.6s
Epoch [30/50]  TrainLoss=2.8635  ValLoss=2.7110  Acc@1=49.24%  Acc@5=75.17%  lr=0.00104  t=28.1s
Epoch [31/50]  TrainLoss=2.9018  ValLoss=2.7175  Acc@1=49.62%  Acc@5=75.34%  lr=0.00095  t=28.2s
Epoch [32/50]  TrainLoss=2.7952  ValLoss=2.6600  Acc@1=50.50%  Acc@5=75.69%  lr=0.00086  t=26.7s
Epoch [33/50]  TrainLoss=2.8231  ValLoss=2.6465  Acc@1=50.67%  Acc@5=76.11%  lr=0.00078  t=26.7s
Epoch [34/50]  TrainLoss=2.7611  ValLoss=2.6420  Acc@1=50.89%  Acc@5=76.13%  lr=0.00070  t=26.5s
Epoch [35/50]  TrainLoss=2.7085  ValLoss=2.6141  Acc@1=51.10%  Acc@5=76.77%  lr=0.00062  t=26.5s
Epoch [36/50]  TrainLoss=2.7877  ValLoss=2.6815  Acc@1=50.49%  Acc@5=75.73%  lr=0.00054  t=26.7s
Epoch [37/50]  TrainLoss=2.7367  ValLoss=2.6069  Acc@1=51.83%  Acc@5=76.74%  lr=0.00047  t=26.6s
Epoch [38/50]  TrainLoss=2.7574  ValLoss=2.6802  Acc@1=51.01%  Acc@5=75.67%  lr=0.00041  t=28.3s
Epoch [39/50]  TrainLoss=2.7172  ValLoss=2.6410  Acc@1=51.41%  Acc@5=76.41%  lr=0.00034  t=27.9s
Epoch [40/50]  TrainLoss=2.7275  ValLoss=2.6357  Acc@1=51.78%  Acc@5=76.83%  lr=0.00029  t=28.3s
Epoch [41/50]  TrainLoss=2.6355  ValLoss=2.6370  Acc@1=51.14%  Acc@5=76.48%  lr=0.00023  t=28.5s
Epoch [42/50]  TrainLoss=2.6411  ValLoss=2.6298  Acc@1=51.68%  Acc@5=76.47%  lr=0.00019  t=26.5s
Epoch [43/50]  TrainLoss=2.6452  ValLoss=2.6120  Acc@1=52.11%  Acc@5=76.65%  lr=0.00014  t=26.9s
Epoch [44/50]  TrainLoss=2.6592  ValLoss=2.6157  Acc@1=52.15%  Acc@5=76.98%  lr=0.00011  t=26.8s
Epoch [45/50]  TrainLoss=2.6497  ValLoss=2.6285  Acc@1=52.54%  Acc@5=76.53%  lr=0.00007  t=26.9s
Epoch [46/50]  TrainLoss=2.6963  ValLoss=2.6598  Acc@1=51.60%  Acc@5=76.43%  lr=0.00005  t=28.7s
Epoch [47/50]  TrainLoss=2.6061  ValLoss=2.6254  Acc@1=52.37%  Acc@5=76.32%  lr=0.00003  t=27.6s
Epoch [48/50]  TrainLoss=2.5832  ValLoss=2.6494  Acc@1=51.64%  Acc@5=76.15%  lr=0.00001  t=27.5s
Epoch [49/50]  TrainLoss=2.6129  ValLoss=2.6122  Acc@1=51.73%  Acc@5=76.81%  lr=0.00000  t=28.4s
Epoch [50/50]  TrainLoss=2.5764  ValLoss=2.6121  Acc@1=51.82%  Acc@5=76.70%  lr=0.00000  t=28.3s

✓  Metrics → results/baseline_efficientnet_b0_metrics.json

✓  All models trained.

  Model                   Acc@1    Acc@5  Params(M)   Lat(ms)  Size(MB)
────────────────────────────────────────────────────────────────────────────────
  mobilenetv2             50.20    76.27       2.48      0.86       9.7
  shufflenetv2            50.63    75.29       1.46      1.04       5.7
  efficientnet_b0         52.54    76.98       4.26      1.28      16.6
────────────────────────────────────────────────────────────────────────────────
Saved → baseline_training_curves.png
Saved → baseline_comparison.png


***********************************************************************************************************************************************
***********************************************************************************************************************************************


$ /home/dgxuser15/.venv/bin/python /home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/scripts/hardware-aware.py
Device : cuda
GPU    : Tesla V100-SXM2-32GB
VRAM   : 31.7 GB
Train : 100,000 | Val : 10,000 | Classes : 200
Transforms and Dataset classes defined.
Train batches : 390 | Val batches : 79
DRY_RUN = False
Search space: 7 ops × 20 cells  →  7.98e+16 architectures
All primitive ops defined: Identity, DepthwiseSepConv, MBConv, ShuffleBlock, SEBlock
SuperNet parameters : 10.20 M
[LUT] Building latency lookup table …
  Cell  0/19 done
  Cell  5/19 done
  Cell 10/19 done
  Cell 15/19 done
✓  LUT saved → /raid/home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/results/latency_lut.json
LUT covers 20 cells × 7 ops
Predicted latency (all-identity arch): 0.91 ms
HW_CFG: {
  "supernet_epochs": 50,
  "batch_size": 256,
  "lr": 0.0005,
  "weight_decay": 0.0001,
  "label_smoothing": 0.1,
  "latency_budget": 5.0,
  "lambda_lat": 0.01,
  "grad_clip": 5.0
}

============================================================
  Supernet Training (One-Shot NAS)
============================================================
  Epoch [01/50]  Loss=5.3285  ValAcc@1=0.51%  t=119.4s
  Epoch [02/50]  Loss=5.3135  ValAcc@1=0.52%  t=29.2s
  Epoch [03/50]  Loss=5.2985  ValAcc@1=0.55%  t=29.6s
  Epoch [04/50]  Loss=5.2819  ValAcc@1=0.60%  t=29.2s
  Epoch [05/50]  Loss=5.2673  ValAcc@1=0.42%  t=26.2s
  Epoch [06/50]  Loss=5.2514  ValAcc@1=0.46%  t=26.0s
  Epoch [07/50]  Loss=5.2377  ValAcc@1=0.55%  t=26.8s
  Epoch [08/50]  Loss=5.2246  ValAcc@1=0.59%  t=26.2s
  Epoch [09/50]  Loss=5.2129  ValAcc@1=0.45%  t=27.2s
  Epoch [10/50]  Loss=5.2003  ValAcc@1=0.52%  t=26.1s
  Epoch [11/50]  Loss=5.1922  ValAcc@1=0.44%  t=25.7s
  Epoch [12/50]  Loss=5.1835  ValAcc@1=0.54%  t=25.9s
  Epoch [13/50]  Loss=5.1763  ValAcc@1=0.47%  t=26.1s
  Epoch [14/50]  Loss=5.1655  ValAcc@1=0.55%  t=26.1s
  Epoch [15/50]  Loss=5.1593  ValAcc@1=0.54%  t=26.1s
  Epoch [16/50]  Loss=5.1523  ValAcc@1=0.49%  t=26.1s
  Epoch [17/50]  Loss=5.1476  ValAcc@1=0.53%  t=26.1s
  Epoch [18/50]  Loss=5.1382  ValAcc@1=0.52%  t=26.0s
  Epoch [19/50]  Loss=5.1315  ValAcc@1=0.54%  t=25.9s
  Epoch [20/50]  Loss=5.1281  ValAcc@1=0.55%  t=25.8s
  Epoch [21/50]  Loss=5.1177  ValAcc@1=0.49%  t=26.0s
  Epoch [22/50]  Loss=5.1102  ValAcc@1=0.50%  t=26.0s
  Epoch [23/50]  Loss=5.1011  ValAcc@1=0.46%  t=26.2s
  Epoch [24/50]  Loss=5.0878  ValAcc@1=0.53%  t=26.1s
  Epoch [25/50]  Loss=5.0783  ValAcc@1=0.43%  t=26.2s
  Epoch [26/50]  Loss=5.3322  ValAcc@1=0.43%  t=131.6s
  Epoch [27/50]  Loss=5.2921  ValAcc@1=0.52%  t=30.6s
  Epoch [28/50]  Loss=5.2813  ValAcc@1=0.51%  t=30.2s
  Epoch [29/50]  Loss=5.2783  ValAcc@1=0.54%  t=30.6s
  Epoch [30/50]  Loss=5.2736  ValAcc@1=0.54%  t=30.4s
  Epoch [31/50]  Loss=5.2695  ValAcc@1=0.40%  t=30.8s
  Epoch [32/50]  Loss=5.2670  ValAcc@1=0.62%  t=30.3s
  Epoch [33/50]  Loss=5.2611  ValAcc@1=0.67%  t=30.2s
  Epoch [34/50]  Loss=5.2580  ValAcc@1=0.48%  t=30.8s
  Epoch [35/50]  Loss=5.2555  ValAcc@1=0.44%  t=30.3s
  Epoch [36/50]  Loss=5.2528  ValAcc@1=0.51%  t=30.5s
  Epoch [37/50]  Loss=5.2514  ValAcc@1=0.53%  t=30.6s
  Epoch [38/50]  Loss=5.2469  ValAcc@1=0.48%  t=30.3s
  Epoch [39/50]  Loss=5.2478  ValAcc@1=0.57%  t=30.5s
  Epoch [40/50]  Loss=5.2444  ValAcc@1=0.52%  t=31.0s
  Epoch [41/50]  Loss=5.2468  ValAcc@1=0.55%  t=30.8s
  Epoch [42/50]  Loss=5.2415  ValAcc@1=0.62%  t=30.8s
  Epoch [43/50]  Loss=5.2385  ValAcc@1=0.66%  t=30.7s
  Epoch [44/50]  Loss=5.2404  ValAcc@1=0.58%  t=30.7s
  Epoch [45/50]  Loss=5.2403  ValAcc@1=0.56%  t=30.8s
  Epoch [46/50]  Loss=5.2386  ValAcc@1=0.56%  t=29.9s
  Epoch [47/50]  Loss=5.2398  ValAcc@1=0.48%  t=30.1s
  Epoch [48/50]  Loss=5.2374  ValAcc@1=0.59%  t=30.6s
  Epoch [49/50]  Loss=5.2387  ValAcc@1=0.58%  t=30.8s
  Epoch [50/50]  Loss=5.2401  ValAcc@1=0.69%  t=30.2s
✓  Supernet saved → /raid/home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/models/supernet_final.pth
Saved → supernet_training.png
EVO_CFG: {
  "population": 50,
  "generations": 20,
  "top_k": 10,
  "mutation_p": 0.15,
  "n_eval_batches": 20
}
evaluate_arch() defined.

============================================================
  Evolutionary Architecture Search
============================================================
  Gen [01/20]  Best acc=1.33%  lat=7.43 ms  Evaluated=50 archs
  Gen [02/20]  Best acc=1.33%  lat=7.43 ms  Evaluated=90 archs
  Gen [03/20]  Best acc=1.33%  lat=7.43 ms  Evaluated=130 archs
  Gen [04/20]  Best acc=1.33%  lat=7.43 ms  Evaluated=170 archs
  Gen [05/20]  Best acc=1.52%  lat=8.37 ms  Evaluated=209 archs
  Gen [06/20]  Best acc=1.52%  lat=8.37 ms  Evaluated=248 archs
  Gen [07/20]  Best acc=1.52%  lat=8.37 ms  Evaluated=288 archs
  Gen [08/20]  Best acc=1.52%  lat=8.37 ms  Evaluated=326 archs
  Gen [09/20]  Best acc=1.52%  lat=8.37 ms  Evaluated=364 archs
  Gen [10/20]  Best acc=1.52%  lat=8.37 ms  Evaluated=403 archs
  Gen [11/20]  Best acc=1.52%  lat=8.37 ms  Evaluated=443 archs
  Gen [12/20]  Best acc=1.52%  lat=8.37 ms  Evaluated=483 archs
  Gen [13/20]  Best acc=1.60%  lat=8.82 ms  Evaluated=523 archs
  Gen [14/20]  Best acc=1.60%  lat=8.82 ms  Evaluated=562 archs
  Gen [15/20]  Best acc=1.64%  lat=7.76 ms  Evaluated=602 archs
  Gen [16/20]  Best acc=1.64%  lat=7.76 ms  Evaluated=641 archs
  Gen [17/20]  Best acc=1.64%  lat=7.76 ms  Evaluated=681 archs
  Gen [18/20]  Best acc=1.68%  lat=8.35 ms  Evaluated=721 archs
  Gen [19/20]  Best acc=1.68%  lat=8.35 ms  Evaluated=761 archs
  Gen [20/20]  Best acc=1.68%  lat=8.35 ms  Evaluated=801 archs
✓  Search history saved (801 architectures)
✓  Best arch → /raid/home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/results/best_arch.json
   Accuracy : 1.68%  |  Latency : 8.35 ms
   Ops      : ['mbconv5x5', 'identity', 'identity', 'se_block', 'se_block', 'mbconv5x5', 'shuffle_block', 'identity', 'mbconv3x3', 'shuffle_block', 'se_block', 'mbconv5x5', 'shuffle_block', 'shuffle_block', 'mbconv3x3', 'shuffle_block', 'dwconv3x3', 'mbconv3x3', 'se_block', 'dwconv3x3']
Saved → pareto_front.png


*****************************************************************************************************************************************************************************
*****************************************************************************************************************************************************************************


$ /home/dgxuser15/.venv/bin/python /home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/scripts/nas.py
Device : cuda
GPU    : Tesla V100-SXM2-32GB
Train : 100,000 | Val : 10,000 | Classes : 200
Arch  : [4, 0, 0, 6, 6, 4, 5, 0, 3, 5, 6, 4, 5, 5, 3, 5, 1, 3, 6, 1]
Train batches : 390 | Val batches : 79
All primitive ops defined.
StandaloneNASModel built — 1.50 M parameters
Ops: ['mbconv5x5', 'identity', 'identity', 'se_block', 'se_block', 'mbconv5x5', 'shuffle_block', 'identity', 'mbconv3x3', 'shuffle_block', 'se_block', 'mbconv5x5', 'shuffle_block', 'shuffle_block', 'mbconv3x3', 'shuffle_block', 'dwconv3x3', 'mbconv3x3', 'se_block', 'dwconv3x3']
FT_CFG: {
  "epochs": 50,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "label_smooth": 0.1,
  "mixup_alpha": 0.2,
  "patience": 12,
  "grad_clip": 5.0
}
DRY_RUN = False
  Epoch [001/50]  Loss=4.9767  ValAcc@1=6.63%  ValAcc@5=20.62%  t=96.9s
  Epoch [002/50]  Loss=4.5060  ValAcc@1=9.81%  ValAcc@5=27.06%  t=27.3s
  Epoch [003/50]  Loss=4.2424  ValAcc@1=11.09%  ValAcc@5=30.44%  t=27.3s
  Epoch [004/50]  Loss=4.0903  ValAcc@1=14.04%  ValAcc@5=35.35%  t=27.6s
  Epoch [005/50]  Loss=3.9485  ValAcc@1=17.78%  ValAcc@5=40.88%  t=28.0s
  Epoch [006/50]  Loss=3.8747  ValAcc@1=16.44%  ValAcc@5=38.99%  t=54.5s
  Epoch [007/50]  Loss=3.7593  ValAcc@1=18.64%  ValAcc@5=42.36%  t=54.4s
  Epoch [008/50]  Loss=3.6607  ValAcc@1=25.07%  ValAcc@5=51.57%  t=47.9s
  Epoch [009/50]  Loss=3.5718  ValAcc@1=19.88%  ValAcc@5=43.98%  t=38.8s
  Epoch [010/50]  Loss=3.5294  ValAcc@1=27.27%  ValAcc@5=54.31%  t=37.8s
  Epoch [011/50]  Loss=3.4724  ValAcc@1=26.56%  ValAcc@5=53.34%  t=54.7s
  Epoch [012/50]  Loss=3.4462  ValAcc@1=27.69%  ValAcc@5=54.41%  t=55.0s
  Epoch [013/50]  Loss=3.4485  ValAcc@1=27.92%  ValAcc@5=54.80%  t=48.3s
  Epoch [014/50]  Loss=3.3075  ValAcc@1=27.55%  ValAcc@5=53.89%  t=39.2s
  Epoch [015/50]  Loss=3.2780  ValAcc@1=27.71%  ValAcc@5=54.04%  t=39.0s
  Epoch [016/50]  Loss=3.2856  ValAcc@1=29.68%  ValAcc@5=55.64%  t=32.1s
  Epoch [017/50]  Loss=3.1956  ValAcc@1=27.64%  ValAcc@5=54.03%  t=27.7s
  Epoch [018/50]  Loss=3.2177  ValAcc@1=32.62%  ValAcc@5=60.02%  t=27.5s
  Epoch [019/50]  Loss=3.1639  ValAcc@1=32.98%  ValAcc@5=60.52%  t=27.5s
  Epoch [020/50]  Loss=3.1217  ValAcc@1=32.27%  ValAcc@5=59.32%  t=27.2s
  Epoch [021/50]  Loss=3.1473  ValAcc@1=35.27%  ValAcc@5=61.96%  t=27.3s
  Epoch [022/50]  Loss=3.1240  ValAcc@1=33.73%  ValAcc@5=61.01%  t=27.7s
  Epoch [023/50]  Loss=3.1050  ValAcc@1=34.25%  ValAcc@5=61.66%  t=27.5s
  Epoch [024/50]  Loss=3.0265  ValAcc@1=33.07%  ValAcc@5=59.59%  t=27.2s
  Epoch [025/50]  Loss=3.0013  ValAcc@1=35.43%  ValAcc@5=62.41%  t=27.0s
  Epoch [026/50]  Loss=2.9714  ValAcc@1=34.23%  ValAcc@5=61.03%  t=27.4s
  Epoch [027/50]  Loss=2.9329  ValAcc@1=35.17%  ValAcc@5=61.61%  t=27.3s
  Epoch [028/50]  Loss=2.9794  ValAcc@1=36.85%  ValAcc@5=63.64%  t=27.1s
  Epoch [029/50]  Loss=2.8540  ValAcc@1=36.09%  ValAcc@5=62.90%  t=27.2s
  Epoch [030/50]  Loss=2.8689  ValAcc@1=37.76%  ValAcc@5=64.04%  t=27.5s
  Epoch [031/50]  Loss=2.9161  ValAcc@1=35.23%  ValAcc@5=62.19%  t=27.1s
  Epoch [032/50]  Loss=2.9106  ValAcc@1=36.55%  ValAcc@5=63.14%  t=27.1s
  Epoch [033/50]  Loss=2.8269  ValAcc@1=35.96%  ValAcc@5=62.97%  t=27.1s
  Epoch [034/50]  Loss=2.8122  ValAcc@1=36.02%  ValAcc@5=63.05%  t=27.2s
  Epoch [035/50]  Loss=2.7283  ValAcc@1=36.76%  ValAcc@5=63.27%  t=27.2s
  Epoch [036/50]  Loss=2.7960  ValAcc@1=38.19%  ValAcc@5=65.60%  t=27.3s
  Epoch [037/50]  Loss=2.7758  ValAcc@1=37.58%  ValAcc@5=64.89%  t=27.5s
  Epoch [038/50]  Loss=2.7411  ValAcc@1=37.54%  ValAcc@5=64.46%  t=27.2s
  Epoch [039/50]  Loss=2.6423  ValAcc@1=37.61%  ValAcc@5=64.44%  t=27.2s
  Epoch [040/50]  Loss=2.7422  ValAcc@1=38.93%  ValAcc@5=65.75%  t=27.3s
  Epoch [041/50]  Loss=2.7211  ValAcc@1=38.72%  ValAcc@5=65.71%  t=27.3s
  Epoch [042/50]  Loss=2.7258  ValAcc@1=38.78%  ValAcc@5=65.16%  t=27.4s
  Epoch [043/50]  Loss=2.7805  ValAcc@1=38.82%  ValAcc@5=65.81%  t=27.2s
  Epoch [044/50]  Loss=2.6797  ValAcc@1=37.75%  ValAcc@5=64.71%  t=27.1s
  Epoch [045/50]  Loss=2.6531  ValAcc@1=38.59%  ValAcc@5=65.45%  t=27.5s
  Epoch [046/50]  Loss=2.7609  ValAcc@1=38.43%  ValAcc@5=64.84%  t=27.3s
  Epoch [047/50]  Loss=2.6687  ValAcc@1=38.69%  ValAcc@5=65.29%  t=27.2s
  Epoch [048/50]  Loss=2.7102  ValAcc@1=37.85%  ValAcc@5=64.36%  t=27.1s
  Epoch [049/50]  Loss=2.5726  ValAcc@1=38.79%  ValAcc@5=65.28%  t=27.3s
  Epoch [050/50]  Loss=2.6252  ValAcc@1=38.75%  ValAcc@5=65.36%  t=27.2s

Best val Acc@1 : 38.93%
Saved → /raid/home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/models/nas_best_finetuned.pth
Saved → nas_finetuning_curves.png

── NAS Final Summary ──────────────────────────────────────
{
  "arch": [
    4,
    0,
    0,
    6,
    6,
    4,
    5,
    0,
    3,
    5,
    6,
    4,
    5,
    5,
    3,
    5,
    1,
    3,
    6,
    1
  ],
  "op_names": [
    "mbconv5x5",
    "identity",
    "identity",
    "se_block",
    "se_block",
    "mbconv5x5",
    "shuffle_block",
    "identity",
    "mbconv3x3",
    "shuffle_block",
    "se_block",
    "mbconv5x5",
    "shuffle_block",
    "shuffle_block",
    "mbconv3x3",
    "shuffle_block",
    "dwconv3x3",
    "mbconv3x3",
    "se_block",
    "dwconv3x3"
  ],
  "params_M": 1.502,
  "best_acc1": 38.93,
  "latency_ms": 7.86,
  "epochs_run": 50
}
Saved → nas_final_summary.json


********************************************************************************************************************************************************************
********************************************************************************************************************************************************************


$ /home/dgxuser15/.venv/bin/python /home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/scripts/evaluation.py
Device : cuda
GPU    : Tesla V100-SXM2-32GB
TF32   : matmul=True  cudnn=True
Val samples : 10,000  |  Classes : 200

=======================================================
  Evaluating : MOBILENETV2
=======================================================
Traceback (most recent call last):
  File "/home/dgxuser15/hw-nas/adaptive-edge-nas-main/adaptive-edge-nas-main/scripts/evaluation.py", line 264, in <module>
    model.load_state_dict(ckpt['model_state'])
  File "/home/dgxuser15/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 2635, in load_state_dict
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for MobileNetV2:
        Missing key(s) in state_dict: "features.0.0.weight", "features.0.1.weight", "features.0.1.bias", "features.0.1.running_mean", "features.0.1.running_var", "features.1.conv.0.0.weight", "features.1.conv.0.1.weight", "features.1.conv.0.1.bias", "features.1.conv.0.1.running_mean", "features.1.conv.0.1.running_var", "features.1.conv.1.weight", "features.1.conv.2.weight", "features.1.conv.2.bias", "features.1.conv.2.running_mean", "features.1.conv.2.running_var", "features.2.conv.0.0.weight", "features.2.conv.0.1.weight", "features.2.conv.0.1.bias", "features.2.conv.0.1.running_mean", "features.2.conv.0.1.running_var", "features.2.conv.1.0.weight", "features.2.conv.1.1.weight", "features.2.conv.1.1.bias", "features.2.conv.1.1.running_mean", "features.2.conv.1.1.running_var", "features.2.conv.2.weight", "features.2.conv.3.weight", "features.2.conv.3.bias", "features.2.conv.3.running_mean", "features.2.conv.3.running_var", "features.3.conv.0.0.weight", "features.3.conv.0.1.weight", "features.3.conv.0.1.bias", "features.3.conv.0.1.running_mean", "features.3.conv.0.1.running_var", "features.3.conv.1.0.weight", "features.3.conv.1.1.weight", "features.3.conv.1.1.bias", "features.3.conv.1.1.running_mean", "features.3.conv.1.1.running_var", "features.3.conv.2.weight", "features.3.conv.3.weight", "features.3.conv.3.bias", "features.3.conv.3.running_mean", "features.3.conv.3.running_var", "features.4.conv.0.0.weight", "features.4.conv.0.1.weight", "features.4.conv.0.1.bias", "features.4.conv.0.1.running_mean", "features.4.conv.0.1.running_var", "features.4.conv.1.0.weight", "features.4.conv.1.1.weight", "features.4.conv.1.1.bias", "features.4.conv.1.1.running_mean", "features.4.conv.1.1.running_var", "features.4.conv.2.weight", "features.4.conv.3.weight", "features.4.conv.3.bias", "features.4.conv.3.running_mean", "features.4.conv.3.running_var", "features.5.conv.0.0.weight", "features.5.conv.0.1.weight", "features.5.conv.0.1.bias", "features.5.conv.0.1.running_mean", "features.5.conv.0.1.running_var", "features.5.conv.1.0.weight", "features.5.conv.1.1.weight", "features.5.conv.1.1.bias", "features.5.conv.1.1.running_mean", "features.5.conv.1.1.running_var", "features.5.conv.2.weight", "features.5.conv.3.weight", "features.5.conv.3.bias", "features.5.conv.3.running_mean", "features.5.conv.3.running_var", "features.6.conv.0.0.weight", "features.6.conv.0.1.weight", "features.6.conv.0.1.bias", "features.6.conv.0.1.running_mean", "features.6.conv.0.1.running_var", "features.6.conv.1.0.weight", "features.6.conv.1.1.weight", "features.6.conv.1.1.bias", "features.6.conv.1.1.running_mean", "features.6.conv.1.1.running_var", "features.6.conv.2.weight", "features.6.conv.3.weight", "features.6.conv.3.bias", "features.6.conv.3.running_mean", "features.6.conv.3.running_var", "features.7.conv.0.0.weight", "features.7.conv.0.1.weight", "features.7.conv.0.1.bias", "features.7.conv.0.1.running_mean", "features.7.conv.0.1.running_var", "features.7.conv.1.0.weight", "features.7.conv.1.1.weight", "features.7.conv.1.1.bias", "features.7.conv.1.1.running_mean", "features.7.conv.1.1.running_var", "features.7.conv.2.weight", "features.7.conv.3.weight", "features.7.conv.3.bias", "features.7.conv.3.running_mean", "features.7.conv.3.running_var", "features.8.conv.0.0.weight", "features.8.conv.0.1.weight", "features.8.conv.0.1.bias", "features.8.conv.0.1.running_mean", "features.8.conv.0.1.running_var", "features.8.conv.1.0.weight", "features.8.conv.1.1.weight", "features.8.conv.1.1.bias", "features.8.conv.1.1.running_mean", "features.8.conv.1.1.running_var", "features.8.conv.2.weight", "features.8.conv.3.weight", "features.8.conv.3.bias", "features.8.conv.3.running_mean", "features.8.conv.3.running_var", "features.9.conv.0.0.weight", "features.9.conv.0.1.weight", "features.9.conv.0.1.bias", "features.9.conv.0.1.running_mean", "features.9.conv.0.1.running_var", "features.9.conv.1.0.weight", "features.9.conv.1.1.weight", "features.9.conv.1.1.bias", "features.9.conv.1.1.running_mean", "features.9.conv.1.1.running_var", "features.9.conv.2.weight", "features.9.conv.3.weight", "features.9.conv.3.bias", "features.9.conv.3.running_mean", "features.9.conv.3.running_var", "features.10.conv.0.0.weight", "features.10.conv.0.1.weight", "features.10.conv.0.1.bias", "features.10.conv.0.1.running_mean", "features.10.conv.0.1.running_var", "features.10.conv.1.0.weight", "features.10.conv.1.1.weight", "features.10.conv.1.1.bias", "features.10.conv.1.1.running_mean", "features.10.conv.1.1.running_var", "features.10.conv.2.weight", "features.10.conv.3.weight", "features.10.conv.3.bias", "features.10.conv.3.running_mean", "features.10.conv.3.running_var", "features.11.conv.0.0.weight", "features.11.conv.0.1.weight", "features.11.conv.0.1.bias", "features.11.conv.0.1.running_mean", "features.11.conv.0.1.running_var", "features.11.conv.1.0.weight", "features.11.conv.1.1.weight", "features.11.conv.1.1.bias", "features.11.conv.1.1.running_mean", "features.11.conv.1.1.running_var", "features.11.conv.2.weight", "features.11.conv.3.weight", "features.11.conv.3.bias", "features.11.conv.3.running_mean", "features.11.conv.3.running_var", "features.12.conv.0.0.weight", "features.12.conv.0.1.weight", "features.12.conv.0.1.bias", "features.12.conv.0.1.running_mean", "features.12.conv.0.1.running_var", "features.12.conv.1.0.weight", "features.12.conv.1.1.weight", "features.12.conv.1.1.bias", "features.12.conv.1.1.running_mean", "features.12.conv.1.1.running_var", "features.12.conv.2.weight", "features.12.conv.3.weight", "features.12.conv.3.bias", "features.12.conv.3.running_mean", "features.12.conv.3.running_var", "features.13.conv.0.0.weight", "features.13.conv.0.1.weight", "features.13.conv.0.1.bias", "features.13.conv.0.1.running_mean", "features.13.conv.0.1.running_var", "features.13.conv.1.0.weight", "features.13.conv.1.1.weight", "features.13.conv.1.1.bias", "features.13.conv.1.1.running_mean", "features.13.conv.1.1.running_var", "features.13.conv.2.weight", "features.13.conv.3.weight", "features.13.conv.3.bias", "features.13.conv.3.running_mean", "features.13.conv.3.running_var", "features.14.conv.0.0.weight", "features.14.conv.0.1.weight", "features.14.conv.0.1.bias", "features.14.conv.0.1.running_mean", "features.14.conv.0.1.running_var", "features.14.conv.1.0.weight", "features.14.conv.1.1.weight", "features.14.conv.1.1.bias", "features.14.conv.1.1.running_mean", "features.14.conv.1.1.running_var", "features.14.conv.2.weight", "features.14.conv.3.weight", "features.14.conv.3.bias", "features.14.conv.3.running_mean", "features.14.conv.3.running_var", "features.15.conv.0.0.weight", "features.15.conv.0.1.weight", "features.15.conv.0.1.bias", "features.15.conv.0.1.running_mean", "features.15.conv.0.1.running_var", "features.15.conv.1.0.weight", "features.15.conv.1.1.weight", "features.15.conv.1.1.bias", "features.15.conv.1.1.running_mean", "features.15.conv.1.1.running_var", "features.15.conv.2.weight", "features.15.conv.3.weight", "features.15.conv.3.bias", "features.15.conv.3.running_mean", "features.15.conv.3.running_var", "features.16.conv.0.0.weight", "features.16.conv.0.1.weight", "features.16.conv.0.1.bias", "features.16.conv.0.1.running_mean", "features.16.conv.0.1.running_var", "features.16.conv.1.0.weight", "features.16.conv.1.1.weight", "features.16.conv.1.1.bias", "features.16.conv.1.1.running_mean", "features.16.conv.1.1.running_var", "features.16.conv.2.weight", "features.16.conv.3.weight", "features.16.conv.3.bias", "features.16.conv.3.running_mean", "features.16.conv.3.running_var", "features.17.conv.0.0.weight", "features.17.conv.0.1.weight", "features.17.conv.0.1.bias", "features.17.conv.0.1.running_mean", "features.17.conv.0.1.running_var", "features.17.conv.1.0.weight", "features.17.conv.1.1.weight", "features.17.conv.1.1.bias", "features.17.conv.1.1.running_mean", "features.17.conv.1.1.running_var", "features.17.conv.2.weight", "features.17.conv.3.weight", "features.17.conv.3.bias", "features.17.conv.3.running_mean", "features.17.conv.3.running_var", "features.18.0.weight", "features.18.1.weight", "features.18.1.bias", "features.18.1.running_mean", "features.18.1.running_var", "classifier.1.weight", "classifier.1.bias". 
        Unexpected key(s) in state_dict: "_orig_mod.features.0.0.weight", "_orig_mod.features.0.1.weight", "_orig_mod.features.0.1.bias", "_orig_mod.features.0.1.running_mean", "_orig_mod.features.0.1.running_var", "_orig_mod.features.0.1.num_batches_tracked", "_orig_mod.features.1.conv.0.0.weight", "_orig_mod.features.1.conv.0.1.weight", "_orig_mod.features.1.conv.0.1.bias", "_orig_mod.features.1.conv.0.1.running_mean", "_orig_mod.features.1.conv.0.1.running_var", "_orig_mod.features.1.conv.0.1.num_batches_tracked", "_orig_mod.features.1.conv.1.weight", "_orig_mod.features.1.conv.2.weight", "_orig_mod.features.1.conv.2.bias", "_orig_mod.features.1.conv.2.running_mean", "_orig_mod.features.1.conv.2.running_var", "_orig_mod.features.1.conv.2.num_batches_tracked", "_orig_mod.features.2.conv.0.0.weight", "_orig_mod.features.2.conv.0.1.weight", "_orig_mod.features.2.conv.0.1.bias", "_orig_mod.features.2.conv.0.1.running_mean", "_orig_mod.features.2.conv.0.1.running_var", "_orig_mod.features.2.conv.0.1.num_batches_tracked", "_orig_mod.features.2.conv.1.0.weight", "_orig_mod.features.2.conv.1.1.weight", "_orig_mod.features.2.conv.1.1.bias", "_orig_mod.features.2.conv.1.1.running_mean", "_orig_mod.features.2.conv.1.1.running_var", "_orig_mod.features.2.conv.1.1.num_batches_tracked", "_orig_mod.features.2.conv.2.weight", "_orig_mod.features.2.conv.3.weight", "_orig_mod.features.2.conv.3.bias", "_orig_mod.features.2.conv.3.running_mean", "_orig_mod.features.2.conv.3.running_var", "_orig_mod.features.2.conv.3.num_batches_tracked", "_orig_mod.features.3.conv.0.0.weight", "_orig_mod.features.3.conv.0.1.weight", "_orig_mod.features.3.conv.0.1.bias", "_orig_mod.features.3.conv.0.1.running_mean", "_orig_mod.features.3.conv.0.1.running_var", "_orig_mod.features.3.conv.0.1.num_batches_tracked", "_orig_mod.features.3.conv.1.0.weight", "_orig_mod.features.3.conv.1.1.weight", "_orig_mod.features.3.conv.1.1.bias", "_orig_mod.features.3.conv.1.1.running_mean", "_orig_mod.features.3.conv.1.1.running_var", "_orig_mod.features.3.conv.1.1.num_batches_tracked", "_orig_mod.features.3.conv.2.weight", "_orig_mod.features.3.conv.3.weight", "_orig_mod.features.3.conv.3.bias", "_orig_mod.features.3.conv.3.running_mean", "_orig_mod.features.3.conv.3.running_var", "_orig_mod.features.3.conv.3.num_batches_tracked", "_orig_mod.features.4.conv.0.0.weight", "_orig_mod.features.4.conv.0.1.weight", "_orig_mod.features.4.conv.0.1.bias", "_orig_mod.features.4.conv.0.1.running_mean", "_orig_mod.features.4.conv.0.1.running_var", "_orig_mod.features.4.conv.0.1.num_batches_tracked", "_orig_mod.features.4.conv.1.0.weight", "_orig_mod.features.4.conv.1.1.weight", "_orig_mod.features.4.conv.1.1.bias", "_orig_mod.features.4.conv.1.1.running_mean", "_orig_mod.features.4.conv.1.1.running_var", "_orig_mod.features.4.conv.1.1.num_batches_tracked", "_orig_mod.features.4.conv.2.weight", "_orig_mod.features.4.conv.3.weight", "_orig_mod.features.4.conv.3.bias", "_orig_mod.features.4.conv.3.running_mean", "_orig_mod.features.4.conv.3.running_var", "_orig_mod.features.4.conv.3.num_batches_tracked", "_orig_mod.features.5.conv.0.0.weight", "_orig_mod.features.5.conv.0.1.weight", "_orig_mod.features.5.conv.0.1.bias", "_orig_mod.features.5.conv.0.1.running_mean", "_orig_mod.features.5.conv.0.1.running_var", "_orig_mod.features.5.conv.0.1.num_batches_tracked", "_orig_mod.features.5.conv.1.0.weight", "_orig_mod.features.5.conv.1.1.weight", "_orig_mod.features.5.conv.1.1.bias", "_orig_mod.features.5.conv.1.1.running_mean", "_orig_mod.features.5.conv.1.1.running_var", "_orig_mod.features.5.conv.1.1.num_batches_tracked", "_orig_mod.features.5.conv.2.weight", "_orig_mod.features.5.conv.3.weight", "_orig_mod.features.5.conv.3.bias", "_orig_mod.features.5.conv.3.running_mean", "_orig_mod.features.5.conv.3.running_var", "_orig_mod.features.5.conv.3.num_batches_tracked", "_orig_mod.features.6.conv.0.0.weight", "_orig_mod.features.6.conv.0.1.weight", "_orig_mod.features.6.conv.0.1.bias", "_orig_mod.features.6.conv.0.1.running_mean", "_orig_mod.features.6.conv.0.1.running_var", "_orig_mod.features.6.conv.0.1.num_batches_tracked", "_orig_mod.features.6.conv.1.0.weight", "_orig_mod.features.6.conv.1.1.weight", "_orig_mod.features.6.conv.1.1.bias", "_orig_mod.features.6.conv.1.1.running_mean", "_orig_mod.features.6.conv.1.1.running_var", "_orig_mod.features.6.conv.1.1.num_batches_tracked", "_orig_mod.features.6.conv.2.weight", "_orig_mod.features.6.conv.3.weight", "_orig_mod.features.6.conv.3.bias", "_orig_mod.features.6.conv.3.running_mean", "_orig_mod.features.6.conv.3.running_var", "_orig_mod.features.6.conv.3.num_batches_tracked", "_orig_mod.features.7.conv.0.0.weight", "_orig_mod.features.7.conv.0.1.weight", "_orig_mod.features.7.conv.0.1.bias", "_orig_mod.features.7.conv.0.1.running_mean", "_orig_mod.features.7.conv.0.1.running_var", "_orig_mod.features.7.conv.0.1.num_batches_tracked", "_orig_mod.features.7.conv.1.0.weight", "_orig_mod.features.7.conv.1.1.weight", "_orig_mod.features.7.conv.1.1.bias", "_orig_mod.features.7.conv.1.1.running_mean", "_orig_mod.features.7.conv.1.1.running_var", "_orig_mod.features.7.conv.1.1.num_batches_tracked", "_orig_mod.features.7.conv.2.weight", "_orig_mod.features.7.conv.3.weight", "_orig_mod.features.7.conv.3.bias", "_orig_mod.features.7.conv.3.running_mean", "_orig_mod.features.7.conv.3.running_var", "_orig_mod.features.7.conv.3.num_batches_tracked", "_orig_mod.features.8.conv.0.0.weight", "_orig_mod.features.8.conv.0.1.weight", "_orig_mod.features.8.conv.0.1.bias", "_orig_mod.features.8.conv.0.1.running_mean", "_orig_mod.features.8.conv.0.1.running_var", "_orig_mod.features.8.conv.0.1.num_batches_tracked", "_orig_mod.features.8.conv.1.0.weight", "_orig_mod.features.8.conv.1.1.weight", "_orig_mod.features.8.conv.1.1.bias", "_orig_mod.features.8.conv.1.1.running_mean", "_orig_mod.features.8.conv.1.1.running_var", "_orig_mod.features.8.conv.1.1.num_batches_tracked", "_orig_mod.features.8.conv.2.weight", "_orig_mod.features.8.conv.3.weight", "_orig_mod.features.8.conv.3.bias", "_orig_mod.features.8.conv.3.running_mean", "_orig_mod.features.8.conv.3.running_var", "_orig_mod.features.8.conv.3.num_batches_tracked", "_orig_mod.features.9.conv.0.0.weight", "_orig_mod.features.9.conv.0.1.weight", "_orig_mod.features.9.conv.0.1.bias", "_orig_mod.features.9.conv.0.1.running_mean", "_orig_mod.features.9.conv.0.1.running_var", "_orig_mod.features.9.conv.0.1.num_batches_tracked", "_orig_mod.features.9.conv.1.0.weight", "_orig_mod.features.9.conv.1.1.weight", "_orig_mod.features.9.conv.1.1.bias", "_orig_mod.features.9.conv.1.1.running_mean", "_orig_mod.features.9.conv.1.1.running_var", "_orig_mod.features.9.conv.1.1.num_batches_tracked", "_orig_mod.features.9.conv.2.weight", "_orig_mod.features.9.conv.3.weight", "_orig_mod.features.9.conv.3.bias", "_orig_mod.features.9.conv.3.running_mean", "_orig_mod.features.9.conv.3.running_var", "_orig_mod.features.9.conv.3.num_batches_tracked", "_orig_mod.features.10.conv.0.0.weight", "_orig_mod.features.10.conv.0.1.weight", "_orig_mod.features.10.conv.0.1.bias", "_orig_mod.features.10.conv.0.1.running_mean", "_orig_mod.features.10.conv.0.1.running_var", "_orig_mod.features.10.conv.0.1.num_batches_tracked", "_orig_mod.features.10.conv.1.0.weight", "_orig_mod.features.10.conv.1.1.weight", "_orig_mod.features.10.conv.1.1.bias", "_orig_mod.features.10.conv.1.1.running_mean", "_orig_mod.features.10.conv.1.1.running_var", "_orig_mod.features.10.conv.1.1.num_batches_tracked", "_orig_mod.features.10.conv.2.weight", "_orig_mod.features.10.conv.3.weight", "_orig_mod.features.10.conv.3.bias", "_orig_mod.features.10.conv.3.running_mean", "_orig_mod.features.10.conv.3.running_var", "_orig_mod.features.10.conv.3.num_batches_tracked", "_orig_mod.features.11.conv.0.0.weight", "_orig_mod.features.11.conv.0.1.weight", "_orig_mod.features.11.conv.0.1.bias", "_orig_mod.features.11.conv.0.1.running_mean", "_orig_mod.features.11.conv.0.1.running_var", "_orig_mod.features.11.conv.0.1.num_batches_tracked", "_orig_mod.features.11.conv.1.0.weight", "_orig_mod.features.11.conv.1.1.weight", "_orig_mod.features.11.conv.1.1.bias", "_orig_mod.features.11.conv.1.1.running_mean", "_orig_mod.features.11.conv.1.1.running_var", "_orig_mod.features.11.conv.1.1.num_batches_tracked", "_orig_mod.features.11.conv.2.weight", "_orig_mod.features.11.conv.3.weight", "_orig_mod.features.11.conv.3.bias", "_orig_mod.features.11.conv.3.running_mean", "_orig_mod.features.11.conv.3.running_var", "_orig_mod.features.11.conv.3.num_batches_tracked", "_orig_mod.features.12.conv.0.0.weight", "_orig_mod.features.12.conv.0.1.weight", "_orig_mod.features.12.conv.0.1.bias", "_orig_mod.features.12.conv.0.1.running_mean", "_orig_mod.features.12.conv.0.1.running_var", "_orig_mod.features.12.conv.0.1.num_batches_tracked", "_orig_mod.features.12.conv.1.0.weight", "_orig_mod.features.12.conv.1.1.weight", "_orig_mod.features.12.conv.1.1.bias", "_orig_mod.features.12.conv.1.1.running_mean", "_orig_mod.features.12.conv.1.1.running_var", "_orig_mod.features.12.conv.1.1.num_batches_tracked", "_orig_mod.features.12.conv.2.weight", "_orig_mod.features.12.conv.3.weight", "_orig_mod.features.12.conv.3.bias", "_orig_mod.features.12.conv.3.running_mean", "_orig_mod.features.12.conv.3.running_var", "_orig_mod.features.12.conv.3.num_batches_tracked", "_orig_mod.features.13.conv.0.0.weight", "_orig_mod.features.13.conv.0.1.weight", "_orig_mod.features.13.conv.0.1.bias", "_orig_mod.features.13.conv.0.1.running_mean", "_orig_mod.features.13.conv.0.1.running_var", "_orig_mod.features.13.conv.0.1.num_batches_tracked", "_orig_mod.features.13.conv.1.0.weight", "_orig_mod.features.13.conv.1.1.weight", "_orig_mod.features.13.conv.1.1.bias", "_orig_mod.features.13.conv.1.1.running_mean", "_orig_mod.features.13.conv.1.1.running_var", "_orig_mod.features.13.conv.1.1.num_batches_tracked", "_orig_mod.features.13.conv.2.weight", "_orig_mod.features.13.conv.3.weight", "_orig_mod.features.13.conv.3.bias", "_orig_mod.features.13.conv.3.running_mean", "_orig_mod.features.13.conv.3.running_var", "_orig_mod.features.13.conv.3.num_batches_tracked", "_orig_mod.features.14.conv.0.0.weight", "_orig_mod.features.14.conv.0.1.weight", "_orig_mod.features.14.conv.0.1.bias", "_orig_mod.features.14.conv.0.1.running_mean", "_orig_mod.features.14.conv.0.1.running_var", "_orig_mod.features.14.conv.0.1.num_batches_tracked", "_orig_mod.features.14.conv.1.0.weight", "_orig_mod.features.14.conv.1.1.weight", "_orig_mod.features.14.conv.1.1.bias", "_orig_mod.features.14.conv.1.1.running_mean", "_orig_mod.features.14.conv.1.1.running_var", "_orig_mod.features.14.conv.1.1.num_batches_tracked", "_orig_mod.features.14.conv.2.weight", "_orig_mod.features.14.conv.3.weight", "_orig_mod.features.14.conv.3.bias", "_orig_mod.features.14.conv.3.running_mean", "_orig_mod.features.14.conv.3.running_var", "_orig_mod.features.14.conv.3.num_batches_tracked", "_orig_mod.features.15.conv.0.0.weight", "_orig_mod.features.15.conv.0.1.weight", "_orig_mod.features.15.conv.0.1.bias", "_orig_mod.features.15.conv.0.1.running_mean", "_orig_mod.features.15.conv.0.1.running_var", "_orig_mod.features.15.conv.0.1.num_batches_tracked", "_orig_mod.features.15.conv.1.0.weight", "_orig_mod.features.15.conv.1.1.weight", "_orig_mod.features.15.conv.1.1.bias", "_orig_mod.features.15.conv.1.1.running_mean", "_orig_mod.features.15.conv.1.1.running_var", "_orig_mod.features.15.conv.1.1.num_batches_tracked", "_orig_mod.features.15.conv.2.weight", "_orig_mod.features.15.conv.3.weight", "_orig_mod.features.15.conv.3.bias", "_orig_mod.features.15.conv.3.running_mean", "_orig_mod.features.15.conv.3.running_var", "_orig_mod.features.15.conv.3.num_batches_tracked", "_orig_mod.features.16.conv.0.0.weight", "_orig_mod.features.16.conv.0.1.weight", "_orig_mod.features.16.conv.0.1.bias", "_orig_mod.features.16.conv.0.1.running_mean", "_orig_mod.features.16.conv.0.1.running_var", "_orig_mod.features.16.conv.0.1.num_batches_tracked", "_orig_mod.features.16.conv.1.0.weight", "_orig_mod.features.16.conv.1.1.weight", "_orig_mod.features.16.conv.1.1.bias", "_orig_mod.features.16.conv.1.1.running_mean", "_orig_mod.features.16.conv.1.1.running_var", "_orig_mod.features.16.conv.1.1.num_batches_tracked", "_orig_mod.features.16.conv.2.weight", "_orig_mod.features.16.conv.3.weight", "_orig_mod.features.16.conv.3.bias", "_orig_mod.features.16.conv.3.running_mean", "_orig_mod.features.16.conv.3.running_var", "_orig_mod.features.16.conv.3.num_batches_tracked", "_orig_mod.features.17.conv.0.0.weight", "_orig_mod.features.17.conv.0.1.weight", "_orig_mod.features.17.conv.0.1.bias", "_orig_mod.features.17.conv.0.1.running_mean", "_orig_mod.features.17.conv.0.1.running_var", "_orig_mod.features.17.conv.0.1.num_batches_tracked", "_orig_mod.features.17.conv.1.0.weight", "_orig_mod.features.17.conv.1.1.weight", "_orig_mod.features.17.conv.1.1.bias", "_orig_mod.features.17.conv.1.1.running_mean", "_orig_mod.features.17.conv.1.1.running_var", "_orig_mod.features.17.conv.1.1.num_batches_tracked", "_orig_mod.features.17.conv.2.weight", "_orig_mod.features.17.conv.3.weight", "_orig_mod.features.17.conv.3.bias", "_orig_mod.features.17.conv.3.running_mean", "_orig_mod.features.17.conv.3.running_var", "_orig_mod.features.17.conv.3.num_batches_tracked", "_orig_mod.features.18.0.weight", "_orig_mod.features.18.1.weight", "_orig_mod.features.18.1.bias", "_orig_mod.features.18.1.running_mean", "_orig_mod.features.18.1.running_var", "_orig_mod.features.18.1.num_batches_tracked", "_orig_mod.classifier.1.weight", "_orig_mod.classifier.1.bias". 