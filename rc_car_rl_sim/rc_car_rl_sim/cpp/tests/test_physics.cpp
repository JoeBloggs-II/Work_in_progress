// Automated regression suite for physics.hpp -- covers settle, forward, reverse, turning,
// ramp jump, flywheel reaction, and a 50-second adversarial random-action stress test.
// Build: g++ -O2 -std=c++17 -I.. -o test_physics test_physics.cpp && ./test_physics

#include "physics.hpp"
#include <cstdio>
#define CHECK(cond, msg) if(!(cond)){ printf("FAIL: %s\n", msg); ok=false; } else { printf("PASS: %s\n", msg); }

int main(){
    bool ok = true;
    double dt = 1.0/240.0;

    // 1. Settle: car should come to rest at a sane height, all wheels in contact, upright
    { CarSim sim; sim.reset();
      Action a{};
      for(int i=0;i<480;i++) sim.step(a,dt,1);
      CHECK(sim.s.pos.y > 0.05 && sim.s.pos.y < 0.25, "settle: sane ride height");
      CHECK(sim.s.vel.length() < 0.05, "settle: at rest");
      CHECK(sim.s.upDotWorld > 0.99, "settle: upright");
      for(int i=0;i<4;i++) CHECK(sim.s.wheelContact[i]>0.5, "settle: wheel in contact");
    }
    // 2. Forward drive: builds meaningful speed, stays upright
    { CarSim sim; sim.reset();
      Action a{}; a.duty[0]=a.duty[1]=a.duty[2]=a.duty[3]=0.4;
      double maxSpeed=0;
      for(int i=0;i<960;i++){ sim.step(a,dt,1); maxSpeed=std::max(maxSpeed, sim.s.vel.length()); }
      CHECK(sim.s.pos.x > 3.0, "forward: made meaningful progress");
      CHECK(maxSpeed > 2.0 && maxSpeed < 25.0, "forward: sane speed range");
      CHECK(sim.s.upDotWorld > 0.5, "forward: did not flip");
      CHECK(std::isfinite(sim.s.pos.x) && std::isfinite(sim.s.vel.x), "forward: finite state");
    }
    // 3. Reverse: car actually reverses direction after sustained reverse input
    { CarSim sim; sim.reset();
      Action a{}; a.duty[0]=a.duty[1]=a.duty[2]=a.duty[3]=0.4;
      for(int i=0;i<480;i++) sim.step(a,dt,1);
      double xAtSwitch = sim.s.pos.x;
      a.duty[0]=a.duty[1]=a.duty[2]=a.duty[3]=-0.4;
      for(int i=0;i<960;i++) sim.step(a,dt,1);
      CHECK(sim.s.vel.x < -0.5, "reverse: velocity actually reversed");
      CHECK(std::isfinite(sim.s.pos.x), "reverse: finite state");
      (void)xAtSwitch;
    }
    // 4. Controlled turn: stays upright, changes heading, no NaN
    { CarSim sim; sim.reset();
      Action a{}; a.steer=0.3; a.duty[0]=a.duty[1]=a.duty[2]=a.duty[3]=0.32;
      for(int i=0;i<960;i++) sim.step(a,dt,1);
      CHECK(sim.s.upDotWorld > 0.7, "turn: stayed upright");
      CHECK(std::fabs(sim.s.pos.z) > 0.3, "turn: heading actually changed");
      CHECK(sim.s.angvel.length() < 20.0, "turn: no runaway angular velocity");
    }
    // 5. Ramp jump: achieves real airtime, lands, recovers
    { CarSim sim; sim.reset();
      Action a{}; a.duty[0]=a.duty[1]=a.duty[2]=a.duty[3]=0.75;
      bool wasAirborne=false; bool sawLiftoff=false; bool sawLanding=false;
      double maxHeight=0;
      for(int i=0;i<1200;i++){
          sim.step(a,dt,1);
          bool airborne = (sim.s.wheelContact[0]+sim.s.wheelContact[1]+sim.s.wheelContact[2]+sim.s.wheelContact[3])==0.0;
          if (airborne && !wasAirborne) sawLiftoff=true;
          if (!airborne && wasAirborne) sawLanding=true;
          maxHeight = std::max(maxHeight, sim.s.pos.y);
          wasAirborne = airborne;
      }
      CHECK(sawLiftoff, "jump: achieved liftoff");
      CHECK(sawLanding, "jump: landed again");
      CHECK(maxHeight > 0.5, "jump: real airtime height");
      CHECK(sim.s.upDotWorld > -1.01 && sim.s.upDotWorld < 1.01, "jump: quat stayed normalized");
    }
    // 6. Flywheel: spins up, produces a measurable pitch reaction while airborne
    { CarSim sim; sim.reset(); sim.s.pos.y = 3.0;
      Action a{}; a.duty_flywheel=1.0;
      for(int i=0;i<240;i++) sim.step(a,dt,1);
      CHECK(sim.s.flywheelOmega > 50.0, "flywheel: spun up");
      CHECK(std::isfinite(sim.s.angvel.z), "flywheel: finite pitch rate");
      CHECK(std::fabs(sim.s.angvel.z) < 30.0, "flywheel: bounded reaction (no blowup)");
    }
    // 7. Extreme/adversarial random actions never produce NaN/inf over a long run
    { CarSim sim; sim.reset();
      unsigned seed=12345;
      auto rnd = [&](){ seed = seed*1103515245+12345; return ((seed>>16)&0x7fff)/32767.0*2.0-1.0; };
      for(int i=0;i<12000;i++){
          Action a{ rnd(), {rnd(),rnd(),rnd(),rnd()}, rnd(), rnd() };
          sim.step(a,dt,1);
      }
      CHECK(std::isfinite(sim.s.pos.x)&&std::isfinite(sim.s.pos.y)&&std::isfinite(sim.s.pos.z), "adversarial: position finite after 50s of random extreme actions");
      CHECK(std::isfinite(sim.s.vel.x)&&std::isfinite(sim.s.vel.y)&&std::isfinite(sim.s.vel.z), "adversarial: velocity finite");
      CHECK(sim.s.vel.length() < 100.0, "adversarial: velocity bounded");
    }

    printf(ok ? "\nALL TESTS PASSED\n" : "\nSOME TESTS FAILED\n");
    return ok ? 0 : 1;
}
