import pyopencl as cl


def initialize_opencl():
    platform=cl.get_platforms()[0]
    device =platform.get_devices()[0]
    ctx =cl.Context([device])
    queue=cl.CommandQueue(ctx)

    return ctx,queue

def build_program(ctx, kernal_file_path):
    with open(kernal_file_path, 'r') as f:
        kernal_src= f.read()

    program =cl.Program(ctx,kernal_src).build()
    return program        
