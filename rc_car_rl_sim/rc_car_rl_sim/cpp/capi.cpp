// capi.cpp -- flat C ABI over CarSim + Rasterizer, loaded by Python via ctypes.
// Build: g++ -O2 -shared -fPIC -std=c++17 -o libcarsim.so capi.cpp
#include "physics.hpp"
#include "render.hpp"
#include <vector>
#include <cstring>

struct Handle {
    CarSim sim;
    std::vector<Tri> worldTris;
    Rasterizer* ras = nullptr; // reused across calls -- constructing fresh color+depth buffers
                               // every single frame was a measurable, easily-avoided allocation cost
    int rasW=0, rasH=0;
    Handle(){ buildWorldGeometry(worldTris); }
    ~Handle(){ delete ras; }

    Rasterizer& getRasterizer(int w, int h){
        if (!ras || rasW!=w || rasH!=h){
            delete ras;
            ras = new Rasterizer(w,h);
            rasW=w; rasH=h;
        }
        return *ras;
    }
};

extern "C" {

void* cs_create(){ return new Handle(); }
void cs_destroy(void* h){ delete static_cast<Handle*>(h); }
void cs_reset(void* h){ static_cast<Handle*>(h)->sim.reset(); }

// action: [steer, duty0,duty1,duty2,duty3, duty_flywheel, cam_pan_rate]  (7 doubles)
void cs_step(void* h, const double* action, double dt, int n_substeps){
    Action a;
    a.steer = action[0];
    a.duty[0]=action[1]; a.duty[1]=action[2]; a.duty[2]=action[3]; a.duty[3]=action[4];
    a.duty_flywheel = action[5];
    a.cam_pan_rate = action[6];
    static_cast<Handle*>(h)->sim.step(a, dt, n_substeps);
}

// out: 44 doubles --
// [0:4]=wheelOmega [4]=steerAngle [5]=flywheelOmega [6]=camPanAngle
// [7:11]=suspCompression [11:15]=wheelContact [15:19]=wheelSlip [19:23]=normalLoad
// [23:29]=imuFront(ax,ay,az,gx,gy,gz) [29:35]=imuRear(...)
// [35]=upDotWorld [36]=simTime
// [37:40]=pos (privileged/debug) [40:44]=quat (privileged/debug)
void cs_get_observation(void* h, double* out){
    CarSim& sim = static_cast<Handle*>(h)->sim;
    const CarState& s = sim.s;
    int o=0;
    for(int i=0;i<4;i++) out[o++]=s.wheelOmega[i];
    out[o++]=s.steerAngle;
    out[o++]=s.flywheelOmega;
    out[o++]=s.camPanAngle;
    for(int i=0;i<4;i++) out[o++]=s.suspCompression[i];
    for(int i=0;i<4;i++) out[o++]=s.wheelContact[i];
    for(int i=0;i<4;i++) out[o++]=s.wheelSlip[i];
    for(int i=0;i<4;i++) out[o++]=s.normalLoad[i];

    double imu[6];
    sim.readIMU(Vec3(car::CHASSIS_HALF_X, 0.0, 0.0), imu);
    for(int i=0;i<6;i++) out[o++]=imu[i];
    sim.readIMU(Vec3(-car::CHASSIS_HALF_X, 0.0, 0.0), imu);
    for(int i=0;i<6;i++) out[o++]=imu[i];

    out[o++]=s.upDotWorld;
    out[o++]=s.simTime;
    out[o++]=s.pos.x; out[o++]=s.pos.y; out[o++]=s.pos.z;
    out[o++]=s.quat.x; out[o++]=s.quat.y; out[o++]=s.quat.z; out[o++]=s.quat.w;
}

void cs_render_onboard(void* h, int width, int height, unsigned char* out_rgb){
    Handle* hd = static_cast<Handle*>(h);
    CarState& s = hd->sim.s;
    // Render world and car geometry as two passes sharing one Z-buffer, instead of copying the
    // (much larger, static) world triangle list into a fresh vector every single frame.
    std::vector<Tri> carTris;
    buildCarGeometry(carTris, s);

    Vec3 camLocal(car::CAM_MOUNT_X, car::CAM_MOUNT_Y, 0);
    Quat panQ = Quat::fromAxisAngle(Vec3(0,1,0), s.camPanAngle);
    Camera cam;
    cam.pos = s.pos + s.quat.rotate(camLocal);
    cam.fwd = (s.quat*panQ).rotate(Vec3(1,0,0));
    cam.up  = s.quat.rotate(Vec3(0,1,0));
    cam.fovYRad = 80.0*M_PI/180.0;
    cam.aspect = (double)width/(double)height;

    Rasterizer& ras = hd->getRasterizer(width, height);
    ras.clear(140,180,220);
    ras.render(hd->worldTris, cam);
    ras.render(carTris, cam);
    std::memcpy(out_rgb, ras.color.data(), ras.color.size());
}

void cs_render_custom(void* h, const double* campos3, const double* camfwd3, const double* camup3,
                       double fovYRad, int width, int height, unsigned char* out_rgb){
    Handle* hd = static_cast<Handle*>(h);
    std::vector<Tri> carTris;
    buildCarGeometry(carTris, hd->sim.s);

    Camera cam;
    cam.pos = Vec3(campos3[0],campos3[1],campos3[2]);
    cam.fwd = Vec3(camfwd3[0],camfwd3[1],camfwd3[2]).normalized();
    cam.up  = Vec3(camup3[0],camup3[1],camup3[2]);
    cam.fovYRad = fovYRad;
    cam.aspect = (double)width/(double)height;

    Rasterizer& ras = hd->getRasterizer(width, height);
    ras.clear(140,180,220);
    ras.render(hd->worldTris, cam);
    ras.render(carTris, cam);
    std::memcpy(out_rgb, ras.color.data(), ras.color.size());
}

void cs_set_pose(void* h, double x, double y, double z, double qx, double qy, double qz, double qw){
    CarState& s = static_cast<Handle*>(h)->sim.s;
    s.pos = Vec3(x,y,z);
    s.quat = Quat(qx,qy,qz,qw).normalized();
    s.vel = Vec3(0,0,0);
    s.angvel = Vec3(0,0,0);
}

} // extern "C"
