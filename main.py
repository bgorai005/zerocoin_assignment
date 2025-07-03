from src.image_processor import process_image
 
if __name__=='__main__':
    input_path='images/input1.png'
    output_path='images/output.png'
    max_luminance =5.0

    process_image(input_path,output_path,max_luminance)