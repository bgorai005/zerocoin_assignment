import pyopencl as cl
import numpy as np 
from PIL import Image 
from src.utils import load_image, save_image
from src.opencl_setup import initialize_opencl, build_program


def process_image(input_path, output_path, max_luminance=5.0):
    #load image
    image_np, width, height=load_image(input_path)
    flat_image=image_np.reshape(-1)

    # Opencl setup
    ctx,queue=initialize_opencl()
    program =build_program(ctx, "kernals/filters.cl")
    mf=cl.mem_flags

    #create Opencl buffers
    input_buf = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=flat_image)
    blur_buf =cl.Buffer(ctx, mf.READ_WRITE, size=flat_image.nbytes)
    output_buf =cl.Buffer(ctx,mf.WRITE_ONLY, size=flat_image.nbytes)


    #kernal 1: Gaussian Blur
    kernal1 =program.gaussian_blur
    kernal1.set_args(input_buf,blur_buf,np.int32(width),np.int32(height))
    cl.enqueue_nd_range_kernel(queue,kernal1,(width,height),None)


    #kernal 2: Logarithmic Tone mapping
    kernal2 =program.log_tone_map
    kernal2.set_args(input_buf,output_buf,np.float32(max_luminance),np.int32(width),np.int32(height))
    cl.enqueue_nd_range_kernel(queue,kernal2,(width,height),None)

    #read output
    result =np.empty_like(flat_image)
    cl.enqueue_copy(queue,result,output_buf)
    queue.finish()

    result_image =result.reshape((height,width,4))
    save_image(result_image, output_path)
