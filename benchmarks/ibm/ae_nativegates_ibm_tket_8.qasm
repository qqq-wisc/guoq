OPENQASM 2.0;
include "qelib1.inc";
qreg eval[7];
qreg q[1];
creg meas[8];
u1(pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(9.400234268163183) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u1(7*pi/2) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(0.14726215563701875) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u1(pi/2) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(9.326603190344699) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/2) eval[3];
u1(pi/2) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(7.461282552275758) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi/2) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi/2) eval[4];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(pi/2) eval[5];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
u1(pi/2) eval[6];
u1(pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(2.5740044351731406) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[0],q[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(0.9272952180016121) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[0],q[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[1],q[0];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(1.2870022175865716) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[1],q[0];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[2],q[0];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(0.5675882184166582) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[2],q[0];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
u1(7*pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(11.127542616019355) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[3],q[0];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(1.1351764368333135) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/2) eval[3];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[3],q[0];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(3*pi) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/2) eval[3];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[4],q[0];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(0.8712397799231674) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi/2) eval[4];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[4],q[0];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi/2) eval[4];
u1(3*pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(1.042923012974604) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi/2) q[0];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[5],q[0];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(1.399113093743464) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(pi/2) eval[5];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[5],q[0];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(pi/2) eval[5];
u1(pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[6],q[0];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
u1(0.3433664661028708) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
u1(pi/2) eval[6];
u3(pi,0,pi) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
cx eval[6],q[0];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
u1(pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
u1(9.473865345981723) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
u1(pi/2) eval[6];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[5],eval[6];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(pi/4) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(pi/2) eval[5];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[5],eval[6];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(3.2111243218438377) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(11.064938481832641) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(5.500200586464531) eval[5];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[4],eval[6];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi/8) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi/2) eval[4];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[4],eval[6];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi/2) eval[4];
cx eval[4],eval[5];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi/4) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(pi/2) eval[4];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
cx eval[4],eval[5];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(6.143449678240702) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(7.715590621376128) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(10.200475871891086) eval[4];
u3(pi,0,pi) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[3],eval[6];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/16) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/2) eval[3];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[3],eval[6];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(3*pi/2) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(10.013826583317467) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/2) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/2) eval[3];
cx eval[3],eval[5];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/8) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/2) eval[3];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
cx eval[3],eval[5];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(3*pi) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/2) eval[3];
cx eval[3],eval[4];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/4) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi/2) eval[3];
u3(pi,0,pi) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
cx eval[3],eval[4];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(5.2422917701336145) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(7.142071065908791) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(9.790267716737569) eval[3];
u3(pi,0,pi) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(pi) eval[4];
u3(pi,0,pi) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(3*pi) eval[5];
u3(pi,0,pi) eval[5];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[2],eval[6];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/32) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[2],eval[6];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
cx eval[2],eval[5];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/16) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
cx eval[2],eval[5];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(3*pi/2) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(3.730641276137879) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
cx eval[2],eval[4];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/8) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
u3(pi,0,pi) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
cx eval[2],eval[4];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
cx eval[2],eval[3];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/4) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(pi/2) eval[2];
u3(pi,0,pi) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
cx eval[2],eval[3];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(6.898665015849973) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(11*pi/3) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(5.6677055985092) eval[2];
u3(pi,0,pi) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u3(pi,0,pi) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u3(pi,0,pi) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(3*pi) eval[6];
u3(pi,0,pi) eval[6];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[1],eval[6];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(0.04908738521233912) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[1],eval[6];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(3*pi) eval[1];
u3(pi,0,pi) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
cx eval[1],eval[5];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/32) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
cx eval[1],eval[5];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(3*pi/2) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(3*pi/16) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
cx eval[1],eval[4];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/16) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u3(pi,0,pi) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
cx eval[1],eval[4];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
cx eval[1],eval[3];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/8) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u3(pi,0,pi) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
cx eval[1],eval[3];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
cx eval[1],eval[2];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/4) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u3(pi,0,pi) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
cx eval[1],eval[2];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/2) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi/4) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi) eval[1];
u3(pi,0,pi) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(pi) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u3(pi,0,pi) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u3(pi,0,pi) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u3(pi,0,pi) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
u1(pi) eval[6];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[0],eval[6];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/128) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u3(pi,0,pi) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
cx eval[0],eval[6];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
cx eval[0],eval[5];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/64) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u3(pi,0,pi) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
cx eval[0],eval[5];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(3*pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(0.7363107781851064) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
cx eval[0],eval[4];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/32) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u3(pi,0,pi) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
cx eval[0],eval[4];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
cx eval[0],eval[3];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/16) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u3(pi,0,pi) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
cx eval[0],eval[3];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
cx eval[0],eval[2];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/8) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u3(pi,0,pi) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
cx eval[0],eval[2];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
cx eval[0],eval[1];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/4) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u3(pi,0,pi) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
cx eval[0],eval[1];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(3*pi/2) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(15*pi/4) eval[0];
u1(-pi/2) eval[0];
u2(0,pi) eval[0];
u1(-pi/2) eval[0];
u1(pi/2) eval[0];
u1(7*pi/2) eval[1];
u1(-pi/2) eval[1];
u2(0,pi) eval[1];
u1(-pi/2) eval[1];
u1(7*pi/4) eval[1];
u1(7*pi/2) eval[2];
u1(-pi/2) eval[2];
u2(0,pi) eval[2];
u1(-pi/2) eval[2];
u1(7*pi/8) eval[2];
u1(7*pi/2) eval[3];
u1(-pi/2) eval[3];
u2(0,pi) eval[3];
u1(-pi/2) eval[3];
u1(7*pi/16) eval[3];
u1(7*pi/2) eval[4];
u1(-pi/2) eval[4];
u2(0,pi) eval[4];
u1(-pi/2) eval[4];
u1(3.0434178831651093) eval[4];
u1(pi/2) eval[5];
u1(-pi/2) eval[5];
u2(0,pi) eval[5];
u1(-pi/2) eval[5];
u1(3.0925052683774568) eval[5];
u1(pi/2) eval[6];
u1(-pi/2) eval[6];
u2(0,pi) eval[6];
u1(-pi/2) eval[6];
u1(6.258641614573416) eval[6];
u1(pi/2) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(12.223004148256297) q[0];
u1(-pi/2) q[0];
u2(0,pi) q[0];
u1(-pi/2) q[0];
u1(pi) q[0];
