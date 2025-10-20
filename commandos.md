// Generate train-dataset command
python3 generate-captcha.py   --font_dir font   --symbols symbols.txt   --count 50000   --output_dir train-dataset   --width 192   --height 96   --min_length 1   --max_length 6
 
 
// Generate test-dataset 
python3 generate-captcha.py   --font_dir font   --symbols symbols.txt   --count 5000   --output_dir test-dataset   --width 192   --height 96   --min_length 1   --max_length 6
 
 
// Train command 
python train.py --data_dir train-dataset --symbols symbols.txt --output_dir models --width 192 --height 96 --batch_size 32 --epochs 50 --validation_split 0.2
