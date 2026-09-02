// physics.hpp -- from-scratch RC car physics. No external physics engine: the chassis is the
// only true 6-DOF rigid body; wheels and the flywheel are scalar rotational DOFs attached to it
// via a spring-damper suspension (wheels only) and motor torque curves.
//
// This file went through substantial debugging; the fixes below are load-bearing, not
// stylistic, and are documented at each site:
//  - suspension range is [REST-TRAVEL, REST], not centered on REST -- a compression spring's
//    rest length IS its fully-extended position, it can't extend further
//  - impact force is capped (real dampers have flow-rate limits)
//  - a hard positional floor correction backstops the capped force model against tunneling
//  - suspension force is disabled when the chassis is badly inverted (upW.y <= 0.15) rather
//    than "helpfully" pushing an upside-down car further into the ground
//  - the slip/grip kinetic-friction sign uses a dead-zone instead of raw sign() of a
//    near-zero float (which was picking up floating-point noise)
//  - WHEEL_PEAK_TORQUE and the chassis inertia factor were tuned together: full motor reaction
//    torque on all 4 wheels simultaneously (a real effect for independent hub motors) was
//    strong enough, on a bare-box inertia estimate, to wheelie the car past recovery
//  - a global finite/bounds safety net resets to a safe state instead of ever handing an RL
//    caller NaN/inf or an unbounded position
#pragma once
#include "vecmath.hpp"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace car {

constexpr double GRAVITY = 9.81;
constexpr double AIR_DENSITY = 1.225;

// ---- chassis ---- (local X=forward, Y=up, Z=right/lateral)
constexpr double CHASSIS_MASS = 2.8;
constexpr double CHASSIS_HALF_X = 0.21;
constexpr double CHASSIS_HALF_Y = 0.06;
constexpr double CHASSIS_HALF_Z = 0.11;

// ---- wheels ----
constexpr double WHEEL_RADIUS = 0.06;
constexpr double WHEEL_WIDTH  = 0.05;
constexpr double WHEEL_MASS   = 0.09;
constexpr double HALF_WHEELBASE = 0.14;
constexpr double HALF_TRACK     = 0.125;           // wheels sit slightly outside the body (0.22m) for rollover stability
constexpr double SUSP_REST   = 0.06;
constexpr double SUSP_TRAVEL = 0.06;
constexpr double SUSP_STIFFNESS = 2600.0;
constexpr double SUSP_DAMPING   = 220.0;
constexpr double TIRE_MU = 1.35;
constexpr double LATERAL_STIFFNESS = 55.0;
constexpr double I_WHEEL = 0.5*WHEEL_MASS*WHEEL_RADIUS*WHEEL_RADIUS;

constexpr double WHEEL_PEAK_TORQUE = 0.42;   // N*m per driven wheel (see file header re: tuning)
constexpr double WHEEL_FREE_SPIN_OMEGA = 340.0;
constexpr double WHEEL_WINDAGE = 4.0e-6;
constexpr double WHEEL_ROLL_RESIST = 0.03;
constexpr double STEER_MAX = 0.42;
constexpr double STEER_RATE = 4.5;

// ---- flywheel (pitch-axis reaction wheel; spin axis = chassis-local Z) ----
constexpr double FLYWHEEL_RADIUS = 0.04;
constexpr double FLYWHEEL_THICKNESS = 0.02;
constexpr double FLYWHEEL_DENSITY = 7000.0;
constexpr double FLYWHEEL_MASS = FLYWHEEL_DENSITY * M_PI * FLYWHEEL_RADIUS*FLYWHEEL_RADIUS*FLYWHEEL_THICKNESS;
constexpr double I_FLYWHEEL = 0.5*FLYWHEEL_MASS*FLYWHEEL_RADIUS*FLYWHEEL_RADIUS;
constexpr double FLYWHEEL_PEAK_TORQUE = 0.10;
constexpr double FLYWHEEL_FREE_SPIN_OMEGA = 1600.0;
constexpr double FLYWHEEL_WINDAGE = 6.0e-7;
constexpr double FLYWHEEL_MOUNT_Y = 0.02;

// ---- camera pan servo ----
constexpr double CAM_PAN_MAX = 1.65;
constexpr double CAM_PAN_RATE_MAX = 3.0;
constexpr double CAM_MOUNT_X = 0.20;
constexpr double CAM_MOUNT_Y = 0.05;

// ---- aerodynamics ----
constexpr double AERO_CD_FRONT = 0.9;
constexpr double AERO_AREA_FRONT = 0.030;
constexpr double AERO_CD_SIDE = 1.1;
constexpr double AERO_AREA_SIDE = 0.055;
constexpr double AERO_CL = 0.35;
constexpr double AERO_COP_X = 0.05;

// ---- ground / ramp ----
constexpr double RAMP_START = 3.0;
constexpr double RAMP_END   = 4.2;
constexpr double RAMP_HEIGHT = 0.55;

inline void wheelConnectionLocal(int i, Vec3& out){
    static const double sx[4] = { 1, 1,-1,-1};
    static const double sz[4] = {-1, 1,-1, 1};
    out = Vec3(sx[i]*HALF_WHEELBASE, -0.01, sz[i]*HALF_TRACK);
}

inline double groundHeight(double x, double /*z*/){
    if (x < RAMP_START) return 0.0;
    if (x < RAMP_END) return (x-RAMP_START)/(RAMP_END-RAMP_START) * RAMP_HEIGHT;
    return 0.0;
}
inline Vec3 groundNormal(double x, double /*z*/){
    if (x < RAMP_START || x >= RAMP_END) return Vec3(0,1,0);
    double theta = std::atan2(RAMP_HEIGHT, RAMP_END-RAMP_START);
    return Vec3(-std::sin(theta), std::cos(theta), 0);
}

} // namespace car

struct CarState {
    Vec3 pos{0,0.3,0};
    Quat quat = Quat::identity();
    Vec3 vel{0,0,0};
    Vec3 angvel{0,0,0};

    double wheelOmega[4] = {0,0,0,0};
    double steerAngle = 0.0;
    double flywheelOmega = 0.0;
    double camPanAngle = 0.0;

    double prevSuspLen[4] = {car::SUSP_REST,car::SUSP_REST,car::SUSP_REST,car::SUSP_REST};
    double suspCompression[4] = {0,0,0,0};
    double wheelContact[4] = {0,0,0,0};
    double wheelSlip[4] = {0,0,0,0};
    double normalLoad[4] = {0,0,0,0};
    double upDotWorld = 1.0;
    Vec3 lastLinAccelWorld{0,0,0};
    Vec3 lastAngAccelWorld{0,0,0};

    Vec3 prevVel{0,0,0};
    Vec3 prevAngvel{0,0,0};

    double simTime = 0.0;
};

struct Action {
    double steer;
    double duty[4];
    double duty_flywheel;
    double cam_pan_rate;
};

inline double motorTorqueCurve(double duty, double omega, double peakTorque, double freeSpinOmega){
    if (duty == 0.0) return 0.0;
    double dir = duty > 0 ? 1.0 : -1.0;
    double wAbs = std::fabs(omega);
    double headroom = std::max(0.0, 1.0 - wAbs/freeSpinOmega);
    bool sameDir = (omega==0.0) || ((omega>0)==(duty>0));
    double factor = sameDir ? headroom : 1.0;
    return dir * std::fabs(duty) * peakTorque * factor;
}

class CarSim {
public:
    CarState s;

    void reset(){
        s = CarState();
        s.pos = Vec3(0, 0.35, -2.0);
    }

    void step(const Action& a, double dt, int n_substeps){
        for(int i=0;i<n_substeps;i++) substep(a, dt);
    }

    // IMU at a chassis-local offset point: [ax,ay,az,gx,gy,gz] in the sensor's own body frame.
    // Accelerometer reads specific force (coordinate accel minus gravity) like a real MEMS part.
    void readIMU(const Vec3& localOffset, double out[6]) const {
        Vec3 r = s.quat.rotate(localOffset);
        Vec3 gravityVec(0,-car::GRAVITY,0);
        Vec3 pointAccel = s.lastLinAccelWorld
                         + s.lastAngAccelWorld.cross(r)
                         + s.angvel.cross(s.angvel.cross(r));
        Vec3 specificForceWorld = pointAccel - gravityVec;
        Vec3 aBody(
            specificForceWorld.dot(s.quat.rotate(Vec3(1,0,0))),
            specificForceWorld.dot(s.quat.rotate(Vec3(0,1,0))),
            specificForceWorld.dot(s.quat.rotate(Vec3(0,0,1)))
        );
        Vec3 gBody(
            s.angvel.dot(s.quat.rotate(Vec3(1,0,0))),
            s.angvel.dot(s.quat.rotate(Vec3(0,1,0))),
            s.angvel.dot(s.quat.rotate(Vec3(0,0,1)))
        );
        out[0]=aBody.x; out[1]=aBody.y; out[2]=aBody.z;
        out[3]=gBody.x; out[4]=gBody.y; out[5]=gBody.z;
    }

private:
    void substep(const Action& a, double dt){
        s.prevVel = s.vel;
        s.prevAngvel = s.angvel;

        Vec3 totalForce(0,-car::CHASSIS_MASS*car::GRAVITY,0);
        Vec3 totalTorque(0,0,0);

        Vec3 fwdW  = s.quat.rotate(Vec3(1,0,0));
        Vec3 upW   = s.quat.rotate(Vec3(0,1,0));
        Vec3 rightW= s.quat.rotate(Vec3(0,0,1));

        double steerTarget = std::clamp(a.steer, -1.0, 1.0) * car::STEER_MAX;
        double maxStep = car::STEER_RATE*dt;
        double dSteer = std::clamp(steerTarget - s.steerAngle, -maxStep, maxStep);
        s.steerAngle += dSteer;

        s.camPanAngle += std::clamp(a.cam_pan_rate,-1.0,1.0)*car::CAM_PAN_RATE_MAX*dt;
        s.camPanAngle = std::clamp(s.camPanAngle, -car::CAM_PAN_MAX, car::CAM_PAN_MAX);

        Vec3 Lspin(0,0,0);

        for(int i=0;i<4;i++){
            bool isFront = (i<2);
            Vec3 fwdLocal, axleLocal;
            if (isFront){
                double cs=std::cos(s.steerAngle), sn=std::sin(s.steerAngle);
                fwdLocal  = Vec3(cs,0,-sn);
                axleLocal = Vec3(sn,0, cs);
            } else {
                fwdLocal = Vec3(1,0,0);
                axleLocal= Vec3(0,0,1);
            }
            Vec3 fwdWi = s.quat.rotate(fwdLocal);
            Vec3 axW   = s.quat.rotate(axleLocal);

            Vec3 connLocal; car::wheelConnectionLocal(i, connLocal);
            Vec3 connW = s.pos + s.quat.rotate(connLocal);

            double ground = car::groundHeight(connW.x, connW.z);
            double desiredSuspLen = connW.y - ground - car::WHEEL_RADIUS;
            // Valid range is [REST-TRAVEL, REST] -- REST is the spring's natural, zero-load,
            // fully-EXTENDED length; it cannot extend further (no droop strap modeled).
            bool inContact = desiredSuspLen <= car::SUSP_REST;
            double clampedLen = std::clamp(desiredSuspLen, car::SUSP_REST-car::SUSP_TRAVEL, car::SUSP_REST);
            s.wheelContact[i] = inContact ? 1.0 : 0.0;

            double compression = car::SUSP_REST - clampedLen; // always >= 0
            double closingSpeed = (s.prevSuspLen[i] - clampedLen)/std::max(dt,1e-6);
            s.prevSuspLen[i] = clampedLen;
            s.suspCompression[i] = compression;

            Vec3 wheelCenterW = connW - upW*clampedLen;
            Vec3 contactPtW = wheelCenterW - upW*car::WHEEL_RADIUS;

            double normalForceMag = 0.0;
            if (inContact && upW.y > 0.15){
                // upW.y<=0.15: chassis is badly inverted -- pushing along local-up would push
                // further into the ground, which is correct physics for a flipped car but not
                // worth modeling forces for (that's an episode-termination condition instead).
                double cappedClosingSpeed = std::clamp(closingSpeed, -4.0, 4.0);
                double raw = car::SUSP_STIFFNESS*compression + car::SUSP_DAMPING*cappedClosingSpeed;
                double maxForce = 15.0 * (car::CHASSIS_MASS*car::GRAVITY/4.0);
                normalForceMag = std::clamp(raw, 0.0, maxForce);
            }
            s.normalLoad[i] = normalForceMag;

            Vec3 rel = contactPtW - s.pos;
            Vec3 ptVel = s.vel + s.angvel.cross(rel);
            double groundSpeedFwd = ptVel.dot(fwdWi);
            double groundSpeedLat = ptVel.dot(axW);

            double motorT = motorTorqueCurve(a.duty[i], s.wheelOmega[i], car::WHEEL_PEAK_TORQUE, car::WHEEL_FREE_SPIN_OMEGA);
            double windageT = car::WHEEL_WINDAGE * s.wheelOmega[i] * std::fabs(s.wheelOmega[i]);
            double rollT = std::fabs(s.wheelOmega[i])>0.05 ? car::WHEEL_ROLL_RESIST*(s.wheelOmega[i]>0?1:-1) : 0.0;
            double netT = motorT - windageT - rollT;

            double longForce = 0.0;
            if (inContact && normalForceMag>1e-6){
                double noSlipOmega = groundSpeedFwd / car::WHEEL_RADIUS;
                double maxStatic = car::TIRE_MU * normalForceMag * car::WHEEL_RADIUS;
                if (std::fabs(netT) <= maxStatic){
                    s.wheelOmega[i] += (noSlipOmega - s.wheelOmega[i]) * std::min(1.0, dt*35.0);
                    longForce = netT / car::WHEEL_RADIUS;
                } else {
                    // Kinetic friction opposes the relative slide direction sign(omega-noSlip),
                    // but right at the grip/slip boundary that difference is ~0 and its sign is
                    // just floating-point noise -- trusting it unconditionally intermittently
                    // flipped the friction force backward. Below a small dead-zone, fall back
                    // to the commanded torque's own direction instead.
                    double diff = s.wheelOmega[i]-noSlipOmega;
                    double slipSign = (std::fabs(diff) > 1e-3) ? (diff>0?1.0:-1.0) : (netT>=0?1.0:-1.0);
                    double kineticT = maxStatic*0.85*slipSign;
                    s.wheelOmega[i] += (netT-kineticT)/car::I_WHEEL*dt;
                    longForce = kineticT / car::WHEEL_RADIUS;
                }
                s.wheelSlip[i] = (s.wheelOmega[i]*car::WHEEL_RADIUS - groundSpeedFwd) / std::max(std::fabs(groundSpeedFwd),0.6);

                double maxLat = car::TIRE_MU*normalForceMag;
                double latForce = std::clamp(-car::LATERAL_STIFFNESS*groundSpeedLat, -maxLat, maxLat);

                Vec3 forceAtContact = fwdWi*longForce + axW*latForce + upW*normalForceMag;
                totalForce += forceAtContact;
                totalTorque += rel.cross(forceAtContact);
            } else {
                // airborne (or flipped): free-spin under motor torque alone
                s.wheelOmega[i] += (netT/car::I_WHEEL)*dt;
                s.wheelSlip[i] = 0.0;
            }
            s.wheelOmega[i] = std::clamp(s.wheelOmega[i], -700.0, 700.0);

            // motor reaction torque on the chassis (Newton's 3rd law at the axle bearing)
            totalTorque += axW * (-(motorT));
            Lspin += axW * (car::I_WHEEL * s.wheelOmega[i]);
        }

        // ---- flywheel ----
        {
            double motorT = motorTorqueCurve(a.duty_flywheel, s.flywheelOmega, car::FLYWHEEL_PEAK_TORQUE, car::FLYWHEEL_FREE_SPIN_OMEGA);
            double windageT = car::FLYWHEEL_WINDAGE * s.flywheelOmega * std::fabs(s.flywheelOmega);
            double netT = motorT - windageT;
            s.flywheelOmega += (netT/car::I_FLYWHEEL)*dt;
            s.flywheelOmega = std::clamp(s.flywheelOmega, -car::FLYWHEEL_FREE_SPIN_OMEGA*1.05, car::FLYWHEEL_FREE_SPIN_OMEGA*1.05);

            Vec3 flyAxisW = rightW;
            totalTorque += flyAxisW * (-(motorT));
            Lspin += flyAxisW * (car::I_FLYWHEEL * s.flywheelOmega);
        }

        // ---- gyroscopic precession reaction: tau = -omega_chassis x L_spin ----
        totalTorque += (s.angvel.cross(Lspin)) * -1.0;

        // Small roll-rate damping, representing tire-sidewall/scrub and (on a real car)
        // anti-roll-bar effects that 4 independent linear dampers don't fully capture. Without
        // this the chassis rocks quite violently right at the edge of its rollover envelope.
        {
            double rollRate = s.angvel.dot(fwdW);
            totalTorque += fwdW * (-0.9 * rollRate * car::CHASSIS_MASS);
        }

        // ---- aerodynamics ----
        Vec3 v = s.vel;
        double vFwd = v.dot(fwdW), vLat = v.dot(rightW);
        double speed = v.length();
        if (speed > 0.02){
            double dragMag = 0.5*car::AIR_DENSITY*car::AERO_CD_FRONT*car::AERO_AREA_FRONT*vFwd*std::fabs(vFwd);
            double sideMag = 0.5*car::AIR_DENSITY*car::AERO_CD_SIDE*car::AERO_AREA_SIDE*vLat*std::fabs(vLat);
            double liftMag = 0.5*car::AIR_DENSITY*car::AERO_CL*car::AERO_AREA_FRONT*speed*speed;
            Vec3 cop = s.pos + fwdW*car::AERO_COP_X;
            Vec3 aeroForce = fwdW*(-dragMag) + rightW*(-sideMag) + upW*liftMag;
            totalForce += aeroForce;
            totalTorque += (cop - s.pos).cross(aeroForce);
        }

        // ---- integrate chassis (semi-implicit Euler) ----
        Vec3 accel = totalForce * (1.0/car::CHASSIS_MASS);
        s.vel += accel*dt;
        s.pos += s.vel*dt;

        // Chassis rotational inertia as a solid box, scaled up by a factor representing real
        // mass distribution (motor, battery, electronics along the body) that a bare uniform-box
        // approximation understates relative to a real chassis.
        double INERTIA_FACTOR = 2.5;
        double Ixx = INERTIA_FACTOR*car::CHASSIS_MASS*(car::CHASSIS_HALF_Y*car::CHASSIS_HALF_Y + car::CHASSIS_HALF_Z*car::CHASSIS_HALF_Z)*(4.0/12.0);
        double Iyy = INERTIA_FACTOR*car::CHASSIS_MASS*(car::CHASSIS_HALF_X*car::CHASSIS_HALF_X + car::CHASSIS_HALF_Z*car::CHASSIS_HALF_Z)*(4.0/12.0);
        double Izz = INERTIA_FACTOR*car::CHASSIS_MASS*(car::CHASSIS_HALF_X*car::CHASSIS_HALF_X + car::CHASSIS_HALF_Y*car::CHASSIS_HALF_Y)*(4.0/12.0);
        Vec3 torqueBody(
            totalTorque.dot(s.quat.rotate(Vec3(1,0,0))),
            totalTorque.dot(s.quat.rotate(Vec3(0,1,0))),
            totalTorque.dot(s.quat.rotate(Vec3(0,0,1)))
        );
        Vec3 angAccelBody(torqueBody.x/Ixx, torqueBody.y/Iyy, torqueBody.z/Izz);
        Vec3 angAccelWorld = s.quat.rotate(angAccelBody);
        s.angvel += angAccelWorld*dt;
        s.angvel = s.angvel * 0.999;
        double avMag = s.angvel.length();
        if (avMag > 60.0) s.angvel = s.angvel * (60.0/avMag); // defensive ceiling; should never fire in normal use
        s.quat = s.quat.integrate(s.angvel, dt);
        s.lastLinAccelWorld = accel;
        s.lastAngAccelWorld = angAccelWorld;

        // ---- hard positional floor correction (backstop under the soft suspension model) ----
        // The spring-damper model above intentionally caps its force (for stability), which
        // means a sufficiently violent impact can still out-run it and tunnel through the
        // ground within a single timestep. Standard supplementary position correction: push
        // back out along up and kill the offending velocity component. Never fires in normal
        // driving; only guarantees tunneling can't happen regardless of impact speed.
        {
            double worstPenetration = 0.0;
            Vec3 worstUp(0,1,0);
            for(int i=0;i<4;i++){
                Vec3 connLocal; car::wheelConnectionLocal(i, connLocal);
                Vec3 connW = s.pos + s.quat.rotate(connLocal);
                Vec3 upWi = s.quat.rotate(Vec3(0,1,0));
                double ground = car::groundHeight(connW.x, connW.z);
                double penetration = ground - (connW.y - car::WHEEL_RADIUS - (car::SUSP_REST-car::SUSP_TRAVEL));
                if (penetration > worstPenetration){ worstPenetration = penetration; worstUp = upWi; }
            }
            if (worstPenetration > 0.0){
                s.pos += worstUp * worstPenetration;
                double vAlong = s.vel.dot(worstUp);
                if (vAlong < 0) s.vel -= worstUp * vAlong;
            }
        }

        s.simTime += dt;
        s.upDotWorld = s.quat.rotate(Vec3(0,1,0)).y;

        // Absolute last-resort safety net: should never fire given the fixes above, but an RL
        // environment must never hand NaN/inf or an absurd state to a learning agent regardless.
        if (!std::isfinite(s.pos.x)||!std::isfinite(s.pos.y)||!std::isfinite(s.pos.z)||
            !std::isfinite(s.vel.x)||!std::isfinite(s.vel.y)||!std::isfinite(s.vel.z)||
            s.pos.y < -5.0 || s.pos.y > 50.0 || s.vel.length() > 100.0){
            Vec3 keepPos = s.pos; double keepTime = s.simTime;
            s = CarState();
            s.pos = Vec3(keepPos.x, 0.35, keepPos.z);
            s.simTime = keepTime;
        }
    }
};
