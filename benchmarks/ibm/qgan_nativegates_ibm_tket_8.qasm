OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg meas[8];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(7.060267748739871) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(5*pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(3*pi/2) q[0];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(5.911516010781627) q[1];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
cx q[0],q[1];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u1(5*pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(3*pi/2) q[0];
u1(3*pi) q[1];
u3(pi,0,pi) q[1];
u1(5*pi/2) q[1];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(3*pi/2) q[1];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(12.428715055910846) q[2];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(pi) q[2];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
cx q[0],q[2];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u1(5*pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(3*pi/2) q[0];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
cx q[1],q[2];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(pi/2) q[1];
u1(5*pi/2) q[1];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(3*pi/2) q[1];
u1(pi) q[2];
u1(5*pi/2) q[2];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(3*pi/2) q[2];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(7.777764390680089) q[3];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(pi) q[3];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
cx q[0],q[3];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u1(5*pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(3*pi/2) q[0];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
cx q[1],q[3];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(pi/2) q[1];
u1(5*pi/2) q[1];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(3*pi/2) q[1];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
cx q[2],q[3];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(pi/2) q[2];
u1(5*pi/2) q[2];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(3*pi/2) q[2];
u1(pi) q[3];
u1(5*pi/2) q[3];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(3*pi/2) q[3];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
u1(4.0447645456514945) q[4];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
cx q[0],q[4];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u1(5*pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(3*pi/2) q[0];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
cx q[1],q[4];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(pi/2) q[1];
u1(5*pi/2) q[1];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(3*pi/2) q[1];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
cx q[2],q[4];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(pi/2) q[2];
u1(5*pi/2) q[2];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(3*pi/2) q[2];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
cx q[3],q[4];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(pi/2) q[3];
u1(5*pi/2) q[3];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(3*pi/2) q[3];
u1(3*pi) q[4];
u3(pi,0,pi) q[4];
u1(5*pi/2) q[4];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
u1(3*pi/2) q[4];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
u1(9.385433753852253) q[5];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
cx q[0],q[5];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u1(5*pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(3*pi/2) q[0];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
cx q[1],q[5];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(pi/2) q[1];
u1(5*pi/2) q[1];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(3*pi/2) q[1];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
cx q[2],q[5];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(pi/2) q[2];
u1(5*pi/2) q[2];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(3*pi/2) q[2];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
cx q[3],q[5];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(pi/2) q[3];
u1(5*pi/2) q[3];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(3*pi/2) q[3];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
cx q[4],q[5];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
u1(pi/2) q[4];
u1(5*pi/2) q[4];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
u1(3*pi/2) q[4];
u1(3*pi) q[5];
u3(pi,0,pi) q[5];
u1(5*pi/2) q[5];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
u1(3*pi/2) q[5];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
u1(6.94272937926584) q[6];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
u1(pi) q[6];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
cx q[0],q[6];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u1(5*pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(3*pi/2) q[0];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
cx q[1],q[6];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(pi/2) q[1];
u1(5*pi/2) q[1];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(3*pi/2) q[1];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
cx q[2],q[6];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(pi/2) q[2];
u1(5*pi/2) q[2];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(3*pi/2) q[2];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
cx q[3],q[6];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(pi/2) q[3];
u1(5*pi/2) q[3];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(3*pi/2) q[3];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
cx q[4],q[6];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
u1(pi/2) q[4];
u1(5*pi/2) q[4];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
u1(3*pi/2) q[4];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
cx q[5],q[6];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
u1(pi/2) q[5];
u1(5*pi/2) q[5];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
u1(3*pi/2) q[5];
u1(pi) q[6];
u1(5*pi/2) q[6];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
u1(3*pi/2) q[6];
u1(pi) q[7];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
u1(6.145363996435932) q[7];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
cx q[0],q[7];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u1(3.8828954972387035) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi) q[0];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
cx q[1],q[7];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(pi/2) q[1];
u1(4.266230165373354) q[1];
u1(-pi/2) q[1];
u2(0,pi) q[1];
u1(-pi/2) q[1];
u1(pi) q[1];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
cx q[2],q[7];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(pi/2) q[2];
u1(7.77838874240608) q[2];
u1(-pi/2) q[2];
u2(0,pi) q[2];
u1(-pi/2) q[2];
u1(pi) q[2];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
cx q[3],q[7];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(pi/2) q[3];
u1(8.218812742878479) q[3];
u1(-pi/2) q[3];
u2(0,pi) q[3];
u1(-pi/2) q[3];
u1(pi) q[3];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
cx q[4],q[7];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
u1(pi/2) q[4];
u1(8.695452926065357) q[4];
u1(-pi/2) q[4];
u2(0,pi) q[4];
u1(-pi/2) q[4];
u1(pi) q[4];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
cx q[5],q[7];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
u1(pi/2) q[5];
u1(8.31727604376601) q[5];
u1(-pi/2) q[5];
u2(0,pi) q[5];
u1(-pi/2) q[5];
u1(pi) q[5];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
cx q[6],q[7];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
u1(pi/2) q[6];
u1(3.5245041483128556) q[6];
u1(-pi/2) q[6];
u2(0,pi) q[6];
u1(-pi/2) q[6];
u1(pi) q[6];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
u1(10.11389864485359) q[7];
u1(-pi/2) q[7];
u2(0,pi) q[7];
u1(-pi/2) q[7];
