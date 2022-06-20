for lr in 1e-5 3e-5 1e-4 3e-4 1e-3; do
  for k in 0 1 2 3; do
    for pl in 96 192 336 720; do
      for sl in 96 192 336 720; do
        # ETTm2
        python -u run.py \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm2.csv \
          --model_id ETTm2 \
          --model ETSformer \
          --data ETTm2 \
          --features M \
          --seq_len ${sl} \
          --pred_len ${pl} \
          --e_layers 2 \
          --d_layers 2 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --K ${k} \
          --learning_rate ${lr} \
          --itr 3

        # ETTm2 univar
        python -u run.py \
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm2.csv \
          --model_id ETTm2 \
          --model ETSformer \
          --data ETTm2 \
          --features S \
          --seq_len ${sl} \
          --pred_len ${pl} \
          --e_layers 2 \
          --d_layers 2 \
          --enc_in 1 \
          --dec_in 1 \
          --c_out 1 \
          --des 'Exp' \
          --K ${k} \
          --learning_rate ${lr} \
          --itr 3

        # ECL
        python -u run.py \
          --root_path ./dataset/electricity/ \
          --data_path electricity.csv \
          --model_id ECL \
          --model ETSformer \
          --data custom \
          --features M \
          --seq_len ${sl} \
          --pred_len ${pl} \
          --e_layers 2 \
          --d_layers 2 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --K ${k} \
          --learning_rate ${lr} \
          --itr 3

        # Traffic
        python -u run.py \
          --root_path ./dataset/traffic/ \
          --data_path traffic.csv \
          --model_id traffic \
          --model ETSformer \
          --data custom \
          --features M \
          --seq_len ${sl} \
          --pred_len ${pl} \
          --e_layers 2 \
          --d_layers 2 \
          --enc_in 862 \
          --dec_in 862 \
          --c_out 862 \
          --des 'Exp' \
          --K ${k} \
          --learning_rate ${lr} \
          --itr 3

        # Weather
        python -u run.py \
          --root_path ./dataset/weather/ \
          --data_path weather.csv \
          --model_id weather \
          --model ETSformer \
          --data custom \
          --features M \
          --seq_len ${sl} \
          --pred_len ${pl} \
          --e_layers 2 \
          --d_layers 2 \
          --enc_in 21 \
          --dec_in 21 \
          --c_out 21 \
          --des 'Exp' \
          --K ${k} \
          --learning_rate ${lr} \
          --itr 3

        # Exchange
        python -u run.py \
          --root_path ./dataset/exchange_rate/ \
          --data_path exchange_rate.csv \
          --model_id Exchange \
          --model ETSformer \
          --data custom \
          --features M \
          --seq_len ${sl} \
          --pred_len ${pl} \
          --e_layers 2 \
          --d_layers 2 \
          --enc_in 8 \
          --dec_in 8 \
          --c_out 8 \
          --des 'Exp' \
          --K ${k} \
          --learning_rate ${lr} \
          --itr 3

        # Exchange univar
        python -u run.py \
          --root_path ./dataset/exchange_rate/ \
          --data_path exchange_rate.csv \
          --model_id Exchange \
          --model ETSformer \
          --data custom \
          --features S \
          --seq_len ${sl} \
          --pred_len ${pl} \
          --e_layers 2 \
          --d_layers 2 \
          --enc_in 1 \
          --dec_in 1 \
          --c_out 1 \
          --des 'Exp' \
          --K ${k} \
          --learning_rate ${lr} \
          --itr 3
      done
    done

    # ILI
    for pl in 24 36 48 60; do
      for sl in 24 36 48 60; do
        python -u run.py \
          --root_path ./dataset/illness/ \
          --data_path national_illness.csv \
          --model_id ili \
          --model ETSformer \
          --data custom \
          --features M \
          --seq_len ${sl} \
          --pred_len ${pl} \
          --e_layers 2 \
          --d_layers 2 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --K ${k} \
          --learning_rate ${lr} \
          --itr 3
      done
    done
  done
done
