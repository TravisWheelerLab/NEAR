
for i in 7 10 12
do
  checkpoint_path="/xdisk/twheeler/colligan/model_data/SequenceVAE/$i/checkpoints/best_loss_model.ckpt"
  echo $checkpoint_path
  evaluate with checkpoint_path=$checkpoint_path
done
