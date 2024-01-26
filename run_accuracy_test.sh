echo "####################"
echo " denoiser_LSTM_Valetini  "
echo " float_exec_test         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_LSTM_Valetini.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --float_exec_test
echo "####################"
echo " denoiser_LSTM_Valetini  "
echo " fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_LSTM_Valetini.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ 
echo "####################"
echo " denoiser_LSTM_Valetini  "
echo " mixedfp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_LSTM_Valetini.onnx --quant_type mixedfp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ 
echo "####################"
echo " denoiser_LSTM_Valetini  "
echo " mixedne16fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_LSTM_Valetini.onnx --quant_type mixedne16fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ 


echo "####################"
echo " denoiser_GRU_dns.onnx  "
echo " float_exec_test         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_GRU_dns.onnx.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --float_exec_test
echo "####################"
echo " denoiser_GRU_dns.onnx  "
echo " fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_GRU_dns.onnx.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ 
echo "####################"
echo " denoiser_GRU_dns.onnx  "
echo " mixedfp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_GRU_dns.onnx.onnx --quant_type mixedfp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ 
echo "####################"
echo " denoiser_GRU_dns.onnx  "
echo " mixedne16fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/denoiser_GRU_dns.onnx.onnx --quant_type mixedne16fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ 



echo "####################"
echo " tt_denoiser_rank_2  "
echo " float_exec_test         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_2.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --float_exec_test --tensor_train
echo "####################"
echo " tt_denoiser_rank_2  "
echo " fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_2.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train
echo "####################"
echo " tt_denoiser_rank_2  "
echo " mixedfp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_2.onnx --quant_type mixedfp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train

echo "####################"
echo " tt_denoiser_rank_4  "
echo " float_exec_test         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_4.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --float_exec_test --tensor_train
echo "####################"
echo " tt_denoiser_rank_4  "
echo " fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_4.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train
echo "####################"
echo " tt_denoiser_rank_4  "
echo " mixedfp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_4.onnx --quant_type mixedfp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train


echo "####################"
echo " tt_denoiser_rank_8  "
echo " float_exec_test         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_8.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --float_exec_test --tensor_train
echo "####################"
echo " tt_denoiser_rank_8  "
echo " fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_8.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train
echo "####################"
echo " tt_denoiser_rank_8  "
echo " mixedfp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_8.onnx --quant_type mixedfp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train


echo "####################"
echo " tt_denoiser_rank_16  "
echo " float_exec_test         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_16.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --float_exec_test --tensor_train
echo "####################"
echo " tt_denoiser_rank_16  "
echo " fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_16.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train
echo "####################"
echo " tt_denoiser_rank_16  "
echo " mixedfp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_16.onnx --quant_type mixedfp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train

echo "####################"
echo " tt_denoiser_rank_48  "
echo " float_exec_test         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_48.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --float_exec_test --tensor_train
echo "####################"
echo " tt_denoiser_rank_48  "
echo " fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_48.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train
echo "####################"
echo " tt_denoiser_rank_48  "
echo " mixedfp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_48.onnx --quant_type mixedfp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train


echo "####################"
echo " tt_denoiser_rank_80  "
echo " float_exec_test         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_80.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --float_exec_test --tensor_train
echo "####################"
echo " tt_denoiser_rank_80  "
echo " fp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_80.onnx --quant_type fp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train
echo "####################"
echo " tt_denoiser_rank_80  "
echo " mixedfp16         "
echo "####################"
python nntool_scripts/test_nntool_model.py --mode test_dataset --trained_model model/tensor_train/tt_denoiser_rank_80.onnx --quant_type mixedfp16 --noisy_dataset dataset/test/noisy/ --clean_dataset dataset/test/clean/ --tensor_train