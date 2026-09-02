// render.hpp -- minimal software rasterizer (no GPU/OpenGL available in this environment).
// Flat-shaded triangles, one directional light + ambient, Z-buffer, procedural ground checker.
#pragma once
#include "vecmath.hpp"
#include "physics.hpp"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

struct Tri { Vec3 a,b,c; uint8_t r,g,b_,unused; bool isGround=false; };

struct Camera {
    Vec3 pos, fwd, up;
    double fovYRad, aspect, znear=0.05, zfar=200.0;
};

inline void addBox(std::vector<Tri>& tris, const Vec3& center, const Quat& q, double hx, double hy, double hz,
                    uint8_t r, uint8_t g, uint8_t b){
    Vec3 local[8] = {
        {-hx,-hy,-hz},{hx,-hy,-hz},{hx,hy,-hz},{-hx,hy,-hz},
        {-hx,-hy, hz},{hx,-hy, hz},{hx,hy, hz},{-hx,hy, hz}
    };
    Vec3 w[8];
    for(int i=0;i<8;i++) w[i] = center + q.rotate(local[i]);
    int faces[6][4] = {{0,1,2,3},{5,4,7,6},{4,0,3,7},{1,5,6,2},{3,2,6,7},{4,5,1,0}};
    for(int f=0; f<6; f++){
        Tri t1{w[faces[f][0]], w[faces[f][1]], w[faces[f][2]], r,g,b,0};
        Tri t2{w[faces[f][0]], w[faces[f][2]], w[faces[f][3]], r,g,b,0};
        tris.push_back(t1); tris.push_back(t2);
    }
}

// cylinder whose axis is chassis-local Z (lateral) -- matches wheel/flywheel convention
inline void addCylinderZ(std::vector<Tri>& tris, const Vec3& center, const Quat& q, double radius, double halfWidth,
                          uint8_t r, uint8_t g, uint8_t b, int sides=10){
    std::vector<Vec3> ringNeg(sides), ringPos(sides);
    for(int i=0;i<sides;i++){
        double ang = (2.0*M_PI*i)/sides;
        Vec3 local(std::cos(ang)*radius, std::sin(ang)*radius, -halfWidth);
        ringNeg[i] = center + q.rotate(local);
        local.z = halfWidth;
        ringPos[i] = center + q.rotate(local);
    }
    Vec3 capNeg = center + q.rotate(Vec3(0,0,-halfWidth));
    Vec3 capPos = center + q.rotate(Vec3(0,0, halfWidth));
    for(int i=0;i<sides;i++){
        int j=(i+1)%sides;
        tris.push_back({ringNeg[i], ringNeg[j], ringPos[j], r,g,b,0});
        tris.push_back({ringNeg[i], ringPos[j], ringPos[i], r,g,b,0});
        tris.push_back({capNeg, ringNeg[j], ringNeg[i], (uint8_t)(r*0.8),(uint8_t)(g*0.8),(uint8_t)(b*0.8),0});
        tris.push_back({capPos, ringPos[i], ringPos[j], (uint8_t)(r*0.8),(uint8_t)(g*0.8),(uint8_t)(b*0.8),0});
    }
}

inline void buildWorldGeometry(std::vector<Tri>& tris){
    // Ground is tiled into a grid rather than 2 giant triangles: a triangle spanning both in
    // front of and behind the camera was being trivially rejected whole by the near-plane cull
    // below (which rejects a triangle if any vertex fails, since there's no real clipping), so
    // the entire ground was invisible with 2 giant triangles. Many smaller tiles means only the
    // ones actually behind the camera get dropped, not everything.
    // Tile size only affects culling granularity, not visual detail (the checker pattern is
    // computed per-pixel from world position, not per-tile) -- larger tiles means far fewer
    // triangles for identical visual output.
    double G = 120, tile = 12.0;
    int n = (int)(2*G/tile);
    for(int ix=0; ix<n; ix++){
        for(int iz=0; iz<n; iz++){
            double x0 = -G + ix*tile, x1 = x0+tile;
            double z0 = -G + iz*tile, z1 = z0+tile;
            Tri g1{{x0,0,z0},{x1,0,z0},{x1,0,z1}, 90,95,100,0}; g1.isGround=true;
            Tri g2{{x0,0,z0},{x1,0,z1},{x0,0,z1}, 90,95,100,0}; g2.isGround=true;
            tris.push_back(g1); tris.push_back(g2);
        }
    }

    double x0=car::RAMP_START, x1=car::RAMP_END, h=car::RAMP_HEIGHT, halfW=1.4;
    Vec3 r0{x0,0,-halfW}, r1{x1,h,-halfW}, r2{x1,h,halfW}, r3{x0,0,halfW};
    tris.push_back({r0,r1,r2, 130,90,60,0});
    tris.push_back({r0,r2,r3, 130,90,60,0});
    Vec3 rb0{x1,0,-halfW}, rb1{x1,h,-halfW}, rb2{x1,h,halfW}, rb3{x1,0,halfW};
    tris.push_back({rb0,rb1,rb2, 110,75,50,0});
    tris.push_back({rb0,rb2,rb3, 110,75,50,0});

    for(int i=0;i<10;i++){
        double t = i/9.0;
        Vec3 base(-10 + t*40, 0, 2.6 + std::sin(t*8.0)*1.8);
        addCylinderZ(tris, base+Vec3(0,0.12,0), Quat::fromAxisAngle(Vec3(1,0,0), M_PI/2), 0.10, 0.12, 235,90,40,8);
    }
}

inline void buildCarGeometry(std::vector<Tri>& tris, const CarState& s){
    addBox(tris, s.pos, s.quat, car::CHASSIS_HALF_X, car::CHASSIS_HALF_Y, car::CHASSIS_HALF_Z, 235,120,30);
    Vec3 canopyLocal(-0.03, car::CHASSIS_HALF_Y*0.9, 0);
    addBox(tris, s.pos + s.quat.rotate(canopyLocal), s.quat, 0.09, 0.035, car::CHASSIS_HALF_Z*0.75, 25,25,28);

    for(int i=0;i<4;i++){
        Vec3 connLocal; car::wheelConnectionLocal(i, connLocal);
        Vec3 connW = s.pos + s.quat.rotate(connLocal);
        Vec3 upW = s.quat.rotate(Vec3(0,1,0));
        Vec3 wheelCenter = connW - upW*s.prevSuspLen[i];
        bool isFront = i<2;
        Quat steerQ = isFront ? Quat::fromAxisAngle(Vec3(0,1,0), s.steerAngle) : Quat::identity();
        Quat wq = s.quat * steerQ;
        addCylinderZ(tris, wheelCenter, wq, car::WHEEL_RADIUS, car::WHEEL_WIDTH*0.5, 30,30,32, 12);
    }

    Vec3 fwLocal(0, car::CHASSIS_HALF_Y + 0.03, 0);
    addCylinderZ(tris, s.pos + s.quat.rotate(fwLocal), s.quat, car::FLYWHEEL_RADIUS, car::FLYWHEEL_THICKNESS*0.5, 180,50,50, 10);

    Vec3 camLocal(car::CAM_MOUNT_X, car::CAM_MOUNT_Y, 0);
    Quat panQ = Quat::fromAxisAngle(Vec3(0,1,0), s.camPanAngle);
    addBox(tris, s.pos + s.quat.rotate(camLocal), s.quat*panQ, 0.015,0.015,0.02, 40,200,220);
}

class Rasterizer {
public:
    int width, height;
    std::vector<uint8_t> color;
    std::vector<double> depth;

    Rasterizer(int w, int h): width(w), height(h), color(w*h*3), depth(w*h) {}

    void clear(uint8_t r, uint8_t g, uint8_t b){
        for(int i=0;i<width*height;i++){ color[i*3]=r; color[i*3+1]=g; color[i*3+2]=b; depth[i]=1e18; }
    }

    void render(const std::vector<Tri>& tris, const Camera& cam){
        Mat4 view = Mat4::lookAt(cam.pos, cam.fwd, cam.up);
        Mat4 proj = Mat4::perspective(cam.fovYRad, cam.aspect, cam.znear, cam.zfar);
        Mat4 vp = Mat4::mul(proj, view);
        Vec3 lightDir = Vec3(0.4, -1.0, 0.25).normalized();

        for(const auto& t : tris){
            double wA,wB,wC;
            Vec3 ca = vp.mulPoint(t.a, wA);
            Vec3 cb = vp.mulPoint(t.b, wB);
            Vec3 cc = vp.mulPoint(t.c, wC);
            // Reject only if ALL three vertices are behind/at the near plane. This isn't proper
            // near-plane clipping (a triangle straddling the plane gets a distorted edge), but
            // it's cheap and avoids dropping an entire valid triangle just because one corner
            // pokes behind the camera.
            if (wA<=cam.znear && wB<=cam.znear && wC<=cam.znear) continue;
            double swA = wA>cam.znear? wA : cam.znear;
            double swB = wB>cam.znear? wB : cam.znear;
            double swC = wC>cam.znear? wC : cam.znear;

            double ax = (ca.x/swA*0.5+0.5)*width, ay = (1.0-(ca.y/swA*0.5+0.5))*height, az = ca.z/swA;
            double bx = (cb.x/swB*0.5+0.5)*width, by = (1.0-(cb.y/swB*0.5+0.5))*height, bz = cb.z/swB;
            double cx = (cc.x/swC*0.5+0.5)*width, cy = (1.0-(cc.y/swC*0.5+0.5))*height, cz = cc.z/swC;

            double minX = std::max(0.0, std::floor(std::min({ax,bx,cx})));
            double maxX = std::min((double)width-1, std::ceil(std::max({ax,bx,cx})));
            double minY = std::max(0.0, std::floor(std::min({ay,by,cy})));
            double maxY = std::min((double)height-1, std::ceil(std::max({ay,by,cy})));
            if (minX>maxX || minY>maxY) continue;

            double area = (bx-ax)*(cy-ay) - (cx-ax)*(by-ay);
            if (std::fabs(area) < 1e-9) continue;

            Vec3 normal = (t.b-t.a).cross(t.c-t.a).normalized();
            double diffuse = std::max(0.0, -normal.dot(lightDir));
            double shade = 0.45 + 0.55*diffuse;

            for(int py=(int)minY; py<=(int)maxY; py++){
                for(int px=(int)minX; px<=(int)maxX; px++){
                    double w0 = ((bx-px)*(cy-py) - (cx-px)*(by-py)) / area;
                    double w1 = ((cx-px)*(ay-py) - (ax-px)*(cy-py)) / area;
                    double w2 = 1.0-w0-w1;
                    if (w0<0||w1<0||w2<0) continue;
                    double z = w0*az + w1*bz + w2*cz;
                    int idx = py*width+px;
                    if (z >= depth[idx]) continue;
                    depth[idx] = z;

                    uint8_t rr=t.r, gg=t.g, bb=t.b_;
                    if (t.isGround){
                        Vec3 wp = t.a*w0 + t.b*w1 + t.c*w2;
                        bool check = (((int)std::floor(wp.x) + (int)std::floor(wp.z)) & 1) == 0;
                        if (check){ rr=70; gg=75; bb=80; } else { rr=110; gg=115; bb=120; }
                    }
                    color[idx*3+0] = (uint8_t)std::clamp(rr*shade, 0.0, 255.0);
                    color[idx*3+1] = (uint8_t)std::clamp(gg*shade, 0.0, 255.0);
                    color[idx*3+2] = (uint8_t)std::clamp(bb*shade, 0.0, 255.0);
                }
            }
        }
    }
};
