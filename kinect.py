#from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush
from OpenGL.GL import *
from OpenGL.arrays import vbo


import sys
import numpy

import freenect
import frame_convert

import timing
timings = timing.Timing()


class Kinect(object):
    def __init__(self):
        #set up initial conditions
        pos = numpy.ndarray((640*480, 1), dtype=numpy.float32)
        self.pos_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.pos_vbo.bind()
        #same shit, different toilet
        self.col_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
        self.col_vbo.bind()


    def get_depth(self):
        if len(freenect.sync_get_depth()) > 0:
            return frame_convert.pretty_depth_cv(freenect.sync_get_depth()[0])
        return None

    def get_video(self):
        if len(freenect.sync_get_video()) > 0:
            return frame_convert.video_cv(freenect.sync_get_video()[0])
        return None





import pyopencl as cl
class CL(object):
    def __init__(self, *args, **kwargs):
        self.clinit()
        self.loadProgram("calibrate.cl")

        self.timings = timings


    def kinect_particles(self):
        depth = get_depth()
        rgb = get_video()
        print type(depth)
        print dir(depth)


    def loadData(self, pos_vbo, col_vbo):
        mf = cl.mem_flags
        self.pos_vbo = pos_vbo
        self.col_vbo = col_vbo

        self.pos = pos_vbo.data
        self.col = col_vbo.data

        #Setup vertex buffer objects and share them with OpenCL as GLBuffers
        self.pos_vbo.bind()
        self.pos_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffers[0]))
        self.col_vbo.bind()
        self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffers[0]))

        #pure OpenCL arrays
        #self.vel_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel)
        #self.pos_gen_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pos)
        #self.vel_gen_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vel)
        self.queue.finish()

        self.imsize = 640*480
        self.num = self.imsize
        dempty = numpy.ndarray((self.imsize, 1), dtype=numpy.float32)
        rgbempty = numpy.ndarray((self.imsize, 3), dtype=numpy.dtype('b'))

        #temp values from calibrated kinect using librgbd calibration from Nicolas Burrus
        ptd = numpy.array([485.377991, 7.568644, 0.013969, 0.000000, 11.347664, -474.452148, 
                0.024067, 0.000000, -312.743378, -279.984619, -0.999613, 0.000000, 
                -8.489457, 2.428294, 0.009412, 1.000000], dtype=numpy.float32)
        iptd = numpy.array([0.001845, -0.000000, -0.000000, 0.000000, 0.000000, -0.001848, 
                -0.000000, 0.000000, -0.575108, 0.489076, -1.000000, 0.000000, 
                0.000000, -0.000000, -0.000000, 1.000000], dtype=numpy.float32)

        mf = cl.mem_flags
        self.depth_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dempty)
        self.rgb_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rgbempty)
        
        self.pt_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ptd)
        self.ipt_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=iptd)
        
        # set up the list of GL objects to share with opencl
        self.gl_objects = [self.pos_cl, self.col_cl]
        
    @timings
    def execute(self, sub_intervals):
        cl.enqueue_acquire_gl_objects(self.queue, self.gl_objects)

        global_size = (self.num,)
        local_size = None

        # set up the Kernel argument list
        w = numpy.int32(640)
        h = numpy.int32(480)
        kernelargs = (self.pos_cl, 
                      self.col_cl, 
                      self.depth_cl,
                      self.rgb_cl, 
                      self.pt_cl, 
                      self.ipt_cl, 
                      w,
                      h)

    
        for i in xrange(0, sub_intervals):
            self.program.kinect(self.queue, global_size, local_size, *(kernelargs))

        cl.enqueue_release_gl_objects(self.queue, self.gl_objects)
        self.queue.finish()
 

    def clinit(self):
        plats = cl.get_platforms()
        from pyopencl.tools import get_gl_sharing_context_properties
        import sys 
        if sys.platform == "darwin":
            self.ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                             devices=[])
        else:
            self.ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, plats[0])]
                + get_gl_sharing_context_properties(), devices=None)
                
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        #print fstr
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()


    def render(self):
        
        glEnable(GL_POINT_SMOOTH)
        glPointSize(2)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #setup the VBOs
        self.col_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, self.col_vbo)

        self.pos_vbo.bind()
        glVertexPointer(4, GL_FLOAT, 0, self.pos_vbo)

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        #draw the VBOs
        glDrawArrays(GL_POINTS, 0, self.num)

        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

        glDisable(GL_BLEND)
     

