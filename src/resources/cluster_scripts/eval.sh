
for th in 0.70
do
for i in {1..17}
do
  checkpoint_path="/xdisk/twheeler/colligan/model_data/SequenceVAE/$i/checkpoints/best_loss_model.ckpt"
  echo $checkpoint_path
  evaluate with checkpoint_path=$checkpoint_path evaluator_args.hit_filename="hits_sept22_$i.txt" evaluator_args.overwrite=True
done
done
