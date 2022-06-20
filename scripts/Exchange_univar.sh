python -u run.py \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange \
  --model ETSformer \
  --data custom \
  --features S \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --K 0 \
  --learning_rate 1e-3 \
  --itr 1

python -u run.py \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange \
  --model ETSformer \
  --data custom \
  --features S \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --K 0 \
  --learning_rate 1e-3 \
  --itr 1

python -u run.py \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange \
  --model ETSformer \
  --data custom \
  --features S \
  --seq_len 192 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --K 1 \
  --learning_rate 3e-4 \
  --itr 1

python -u run.py \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange \
  --model ETSformer \
  --data custom \
  --features S \
  --seq_len 720 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 2 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --K 1 \
  --learning_rate 3e-5 \
  --itr 1

