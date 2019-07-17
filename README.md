# SRCNN
Implemented with PyTorch

## Sample Usage
	 > python3 srcnn_run.py --input_image <input_image> --scale_factor 3.0 --model <model_name> --cuda --output_filename <output_filename>

## Training with pruning
	 > python3 srcnn_main.py --upscale_factor 3 --batch_size 10 --cuda --test_batch_size 20 --epochs 110 --lr <learning_rate>  --seed 5000

## Quantization
	> python3 weight_sharing.py <model name>
	
## Huffman coding
	> python3 Huffman_encode.py <model name>
