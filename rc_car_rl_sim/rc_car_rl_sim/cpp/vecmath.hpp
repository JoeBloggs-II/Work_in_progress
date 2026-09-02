// vecmath.hpp -- minimal Vec3 / Quat / Mat4 math used by the physics engine and rasterizer.
// Deliberately dependency-free (no Eigen/GLM) since we don't have network access to fetch them.
#pragma once
#include <cmath>
#include <cstring>

struct Vec3 {
    double x=0, y=0, z=0;
    Vec3() {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator-() const { return {-x,-y,-z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    Vec3& operator+=(const Vec3& o){ x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o){ x-=o.x; y-=o.y; z-=o.z; return *this; }
    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const {
        return { y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x };
    }
    double length() const { return std::sqrt(x*x+y*y+z*z); }
    Vec3 normalized() const {
        double L = length();
        if (L < 1e-12) return {0,0,0};
        return {x/L, y/L, z/L};
    }
};
inline Vec3 operator*(double s, const Vec3& v){ return v*s; }

// Hamilton quaternion, (x,y,z,w) with w the scalar part.
struct Quat {
    double x=0, y=0, z=0, w=1;
    Quat() {}
    Quat(double x_, double y_, double z_, double w_) : x(x_), y(y_), z(z_), w(w_) {}

    static Quat identity(){ return Quat(0,0,0,1); }

    static Quat fromAxisAngle(const Vec3& axis, double angle){
        Vec3 a = axis.normalized();
        double s = std::sin(angle*0.5);
        return Quat(a.x*s, a.y*s, a.z*s, std::cos(angle*0.5));
    }

    Quat operator*(const Quat& o) const {
        // this * o  (apply o first, then this)
        return Quat(
            w*o.x + x*o.w + y*o.z - z*o.y,
            w*o.y - x*o.z + y*o.w + z*o.x,
            w*o.z + x*o.y - y*o.x + z*o.w,
            w*o.w - x*o.x - y*o.y - z*o.z
        );
    }

    Vec3 rotate(const Vec3& v) const {
        // v' = q * v * q^-1, expanded (standard formula)
        Vec3 u(x,y,z);
        double s = w;
        Vec3 uv = u.cross(v);
        Vec3 uuv = u.cross(uv);
        return v + (uv*(2.0*s)) + (uuv*2.0);
    }

    Quat normalized() const {
        double L = std::sqrt(x*x+y*y+z*z+w*w);
        if (L < 1e-12) return Quat::identity();
        return Quat(x/L,y/L,z/L,w/L);
    }

    // integrate orientation given world-space angular velocity omega over dt (semi-implicit)
    Quat integrate(const Vec3& omega, double dt) const {
        Quat dq(omega.x*dt*0.5, omega.y*dt*0.5, omega.z*dt*0.5, 0.0);
        Quat r = Quat(x + (dq.w*x + dq.x*w + dq.y*z - dq.z*y),
                       y + (dq.w*y - dq.x*z + dq.y*w + dq.z*x),
                       z + (dq.w*z + dq.x*y - dq.y*x + dq.z*w),
                       w + (dq.w*w - dq.x*x - dq.y*y - dq.z*z));
        return r.normalized();
    }

    // 3x3 rotation matrix, row-major in a flat 9-array
    void toMat3(double m[9]) const {
        double xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
        m[0]=1-2*(yy+zz); m[1]=2*(xy-wz);   m[2]=2*(xz+wy);
        m[3]=2*(xy+wz);   m[4]=1-2*(xx+zz); m[5]=2*(yz-wx);
        m[6]=2*(xz-wy);   m[7]=2*(yz+wx);   m[8]=1-2*(xx+yy);
    }
};

struct Mat4 {
    double m[16]; // column-major (OpenGL-style): m[col*4+row]

    static Mat4 identity(){
        Mat4 r{};
        for(int i=0;i<16;i++) r.m[i]=0;
        r.m[0]=r.m[5]=r.m[10]=r.m[15]=1;
        return r;
    }

    static Mat4 perspective(double fovYRad, double aspect, double zNear, double zFar){
        Mat4 r{}; for(int i=0;i<16;i++) r.m[i]=0;
        double f = 1.0/std::tan(fovYRad*0.5);
        r.m[0] = f/aspect;
        r.m[5] = f;
        r.m[10] = (zFar+zNear)/(zNear-zFar);
        r.m[11] = -1.0;
        r.m[14] = (2*zFar*zNear)/(zNear-zFar);
        return r;
    }

    static Mat4 lookAt(const Vec3& eye, const Vec3& fwd_, const Vec3& up_){
        Vec3 fwd = fwd_.normalized();
        Vec3 right = fwd.cross(up_).normalized();
        Vec3 up = right.cross(fwd);
        Mat4 r = Mat4::identity();
        r.m[0]=right.x; r.m[4]=right.y; r.m[8]=right.z;
        r.m[1]=up.x;    r.m[5]=up.y;    r.m[9]=up.z;
        r.m[2]=-fwd.x;  r.m[6]=-fwd.y;  r.m[10]=-fwd.z;
        r.m[12] = -right.dot(eye);
        r.m[13] = -up.dot(eye);
        r.m[14] = fwd.dot(eye);
        return r;
    }

    Vec3 mulPoint(const Vec3& p, double& outW) const {
        double x = m[0]*p.x + m[4]*p.y + m[8]*p.z  + m[12];
        double y = m[1]*p.x + m[5]*p.y + m[9]*p.z  + m[13];
        double z = m[2]*p.x + m[6]*p.y + m[10]*p.z + m[14];
        double w = m[3]*p.x + m[7]*p.y + m[11]*p.z + m[15];
        outW = w;
        return {x,y,z};
    }

    static Mat4 mul(const Mat4& a, const Mat4& b){
        Mat4 r{};
        for(int c=0;c<4;c++){
            for(int row=0; row<4; row++){
                double s=0;
                for(int k=0;k<4;k++) s += a.m[k*4+row]*b.m[c*4+k];
                r.m[c*4+row]=s;
            }
        }
        return r;
    }
};
