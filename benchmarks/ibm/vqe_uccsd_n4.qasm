OPENQASM 2.0;
include "qelib1.inc";
qreg reg[4];
u2(0,pi) reg[0];
u2(0,pi) reg[0];
u2(0,pi) reg[2];
u2(0,pi) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(2.151746) reg[0];
cx reg[1],reg[0];
u2(0,pi) reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[0];
u3(pi,pi/2,pi/2) reg[0];
cx reg[2],reg[1];
cx reg[3],reg[2];
u2(0,pi) reg[2];
u3(pi,pi/2,pi/2) reg[2];
u2(0,pi) reg[3];
u3(pi,pi/2,pi/2) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(2.151746) reg[0];
cx reg[1],reg[0];
u3(pi,pi/2,pi/2) reg[0];
u3(pi,pi/2,pi/2) reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[0];
cx reg[2],reg[1];
cx reg[3],reg[2];
u3(pi,pi/2,pi/2) reg[2];
u3(pi,pi/2,pi/2) reg[2];
u3(pi,pi/2,pi/2) reg[3];
u2(0,pi) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(2.151746) reg[0];
cx reg[1],reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[0];
u3(pi,pi/2,pi/2) reg[0];
u2(0,pi) reg[0];
cx reg[2],reg[1];
cx reg[3],reg[2];
u3(pi,pi/2,pi/2) reg[2];
u2(0,pi) reg[2];
u2(0,pi) reg[3];
u3(pi,pi/2,pi/2) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(2.151746) reg[0];
cx reg[1],reg[0];
u3(pi,pi/2,pi/2) reg[0];
u2(0,pi) reg[0];
u2(0,pi) reg[0];
u2(0,pi) reg[0];
cx reg[2],reg[1];
cx reg[3],reg[2];
u2(0,pi) reg[2];
u3(pi,pi/2,pi/2) reg[2];
u3(pi,pi/2,pi/2) reg[3];
u3(pi,pi/2,pi/2) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(2.151746) reg[0];
cx reg[1],reg[0];
u2(0,pi) reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[0];
u3(pi,pi/2,pi/2) reg[0];
cx reg[2],reg[1];
cx reg[3],reg[2];
u3(pi,pi/2,pi/2) reg[2];
u2(0,pi) reg[2];
u3(pi,pi/2,pi/2) reg[3];
u2(0,pi) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(2.151746) reg[0];
cx reg[1],reg[0];
u3(pi,pi/2,pi/2) reg[0];
u3(pi,pi/2,pi/2) reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[0];
cx reg[2],reg[1];
cx reg[3],reg[2];
u2(0,pi) reg[2];
u2(0,pi) reg[2];
u2(0,pi) reg[3];
u3(pi,pi/2,pi/2) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(2.151746) reg[0];
cx reg[1],reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[0];
u3(pi,pi/2,pi/2) reg[0];
u2(0,pi) reg[0];
cx reg[2],reg[1];
cx reg[3],reg[2];
u2(0,pi) reg[2];
u3(pi,pi/2,pi/2) reg[2];
u3(pi,pi/2,pi/2) reg[3];
u2(0,pi) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(2.151746) reg[0];
cx reg[1],reg[0];
u3(pi,pi/2,pi/2) reg[0];
u2(0,pi) reg[0];
u2(0,pi) reg[0];
cx reg[2],reg[1];
u2(0,pi) reg[1];
cx reg[1],reg[0];
u1(1.995482) reg[0];
cx reg[1],reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[0];
u2(0,pi) reg[1];
u3(pi,pi/2,pi/2) reg[1];
cx reg[1],reg[0];
u1(1.995482) reg[0];
cx reg[1],reg[0];
u3(pi,pi/2,pi/2) reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[1];
cx reg[3],reg[2];
u3(pi,pi/2,pi/2) reg[2];
u2(0,pi) reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(4.332582) reg[0];
cx reg[1],reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[0];
cx reg[2],reg[1];
u2(0,pi) reg[2];
u3(pi,pi/2,pi/2) reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(4.332582) reg[0];
cx reg[1],reg[0];
u3(pi,pi/2,pi/2) reg[0];
u2(0,pi) reg[0];
cx reg[2],reg[1];
u2(0,pi) reg[1];
u3(pi,pi/2,pi/2) reg[2];
u2(0,pi) reg[2];
cx reg[2],reg[1];
u1(0.461922) reg[1];
cx reg[2],reg[1];
u2(0,pi) reg[1];
u3(pi,pi/2,pi/2) reg[1];
u2(0,pi) reg[2];
u3(pi,pi/2,pi/2) reg[2];
cx reg[2],reg[1];
u1(0.461922) reg[1];
cx reg[2],reg[1];
u3(pi,pi/2,pi/2) reg[1];
u3(pi,pi/2,pi/2) reg[2];
u2(0,pi) reg[3];
u2(0,pi) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(1.086976) reg[0];
cx reg[1],reg[0];
u2(0,pi) reg[0];
u3(pi,pi/2,pi/2) reg[0];
cx reg[2],reg[1];
cx reg[3],reg[2];
u2(0,pi) reg[3];
u3(pi,pi/2,pi/2) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
cx reg[1],reg[0];
u1(1.086976) reg[0];
cx reg[1],reg[0];
u3(pi,pi/2,pi/2) reg[0];
cx reg[2],reg[1];
u2(0,pi) reg[1];
cx reg[3],reg[2];
u3(pi,pi/2,pi/2) reg[3];
u2(0,pi) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
u1(2.258394) reg[1];
cx reg[2],reg[1];
u2(0,pi) reg[1];
u3(pi,pi/2,pi/2) reg[1];
cx reg[3],reg[2];
u2(0,pi) reg[3];
u3(pi,pi/2,pi/2) reg[3];
cx reg[3],reg[2];
cx reg[2],reg[1];
u1(2.258394) reg[1];
cx reg[2],reg[1];
u3(pi,pi/2,pi/2) reg[1];
cx reg[3],reg[2];
u2(0,pi) reg[2];
u3(pi,pi/2,pi/2) reg[3];
u2(0,pi) reg[3];
cx reg[3],reg[2];
u1(1.228531) reg[2];
cx reg[3],reg[2];
u2(0,pi) reg[2];
u3(pi,pi/2,pi/2) reg[2];
u2(0,pi) reg[3];
u3(pi,pi/2,pi/2) reg[3];
cx reg[3],reg[2];
u1(1.228531) reg[2];
cx reg[3],reg[2];
u3(pi,pi/2,pi/2) reg[2];
u3(pi,pi/2,pi/2) reg[3];
