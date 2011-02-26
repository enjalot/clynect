from OpenGL.GL import GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW, glFlush

import pyopencl as cl
import clutil

import sys
import numpy

import freenect
import frame_convert

def get_depth():
    if len(freenect.sync_get_depth()) > 0:
        return frame_convert.pretty_depth_cv(freenect.sync_get_depth()[0])
    return None

def get_video():
    if len(freenect.sync_get_video()) > 0:
        return frame_convert.video_cv(freenect.sync_get_video()[0])
    return None

class Kinect(clutil.CLProgram):

    def kinect_particles():
        depth = get_depth()
        rgb = get_video()
        print type(depth)
        print dir(depth)


    def loadData(self, pos_vbo, col_vbo, vel):
        mf = cl.mem_flags
        self.pos_vbo = pos_vbo
        self.col_vbo = col_vbo

        self.pos = pos_vbo.data
        self.col = col_vbo.data
        self.vel = vel

        #Setup vertex buffer objects and share them with OpenCL as GLBuffers
        self.pos_vbo.bind()
        self.pos_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.pos_vbo.buffers[0]))
        self.col_vbo.bind()
        self.col_cl = cl.GLBuffer(self.ctx, mf.READ_WRITE, int(self.col_vbo.buffers[0]))

        #pure OpenCL arrays
        self.vel_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vel)
        self.pos_gen_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.pos)
        self.vel_gen_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.vel)
        self.queue.finish()

        self.imsize = 640*480
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
        
        # set up the Kernel argument list
        w = numpy.int32(640)
        h = numpy.int32(480)
        self.kernelargs = (self.pos_cl, 
                           self.col_cl, 
                           self.depth_cl,
                           self.rgb_cl, 
                           self.pt_cl, 
                           self.ipt_cl, 
                           w,
                           h)


