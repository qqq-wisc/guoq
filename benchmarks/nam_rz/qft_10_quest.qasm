OPENQASM 2.0;
include "qelib1.inc";
qreg node[10];
creg c[10];
rz(pi/32) node[0];
rz(pi/8) node[1];
rz(pi/2) node[2];
rz(-pi/2) node[2];
h node[2];
rz(-pi/2) node[2];
rz(pi/2) node[2];
rz(pi/16) node[3];
rz(pi/256) node[4];
rz(pi/128) node[5];
rz(pi/64) node[6];
rz(pi/4) node[7];
cx node[7],node[2];
rz(7*pi/4) node[2];
cx node[7],node[2];
rz(pi/4) node[2];
cx node[1],node[2];
rz(5.8904862) node[2];
cx node[1],node[2];
rz(pi/4) node[1];
rz(pi/8) node[2];
cx node[3],node[2];
rz(6.0868358) node[2];
cx node[3],node[2];
rz(pi/16) node[2];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[0],node[1];
rz(6.1850105) node[1];
cx node[0],node[1];
rz(pi/16) node[0];
rz(pi/32) node[1];
rz(pi/8) node[3];
cx node[5],node[0];
cx node[6],node[1];
rz(6.2340979) node[1];
cx node[6],node[1];
rz(pi/64) node[1];
cx node[0],node[1];
cx node[5],node[0];
cx node[0],node[1];
rz(6.2586416) node[1];
cx node[5],node[0];
cx node[0],node[1];
cx node[5],node[0];
cx node[0],node[1];
rz(pi/128) node[1];
rz(pi/64) node[5];
rz(pi/32) node[6];
rz(pi/2) node[7];
rz(-pi/2) node[7];
h node[7];
rz(-pi/2) node[7];
rz(pi/2) node[7];
cx node[2],node[7];
rz(7*pi/4) node[7];
cx node[2],node[7];
rz(pi/2) node[2];
rz(-pi/2) node[2];
h node[2];
rz(-pi/2) node[2];
rz(pi/2) node[2];
cx node[3],node[2];
rz(pi/4) node[7];
cx node[2],node[7];
cx node[3],node[2];
cx node[2],node[7];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[4],node[3];
rz(6.2709135) node[3];
cx node[4],node[3];
rz(pi/256) node[3];
rz(pi/128) node[4];
rz(5.8904862) node[7];
cx node[2],node[7];
rz(pi/4) node[2];
cx node[2],node[1];
rz(7*pi/4) node[1];
cx node[2],node[1];
rz(pi/4) node[1];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
rz(pi/2) node[2];
rz(-pi/2) node[2];
h node[2];
rz(-pi/2) node[2];
rz(pi/2) node[2];
cx node[1],node[2];
rz(pi/8) node[7];
cx node[2],node[7];
cx node[1],node[2];
cx node[2],node[7];
cx node[1],node[2];
rz(6.0868358) node[7];
cx node[2],node[7];
cx node[1],node[2];
rz(pi/8) node[1];
cx node[1],node[0];
rz(5.8904862) node[0];
cx node[1],node[0];
rz(pi/8) node[0];
rz(pi/4) node[1];
cx node[2],node[7];
cx node[1],node[2];
rz(7*pi/4) node[2];
cx node[1],node[2];
rz(pi/2) node[1];
rz(-pi/2) node[1];
h node[1];
rz(-pi/2) node[1];
rz(pi/2) node[1];
cx node[0],node[1];
rz(pi/4) node[2];
rz(pi/16) node[7];
cx node[6],node[7];
rz(6.1850105) node[7];
cx node[6],node[7];
rz(pi/16) node[6];
cx node[1],node[6];
cx node[0],node[1];
rz(6.0868358) node[0];
cx node[1],node[6];
cx node[0],node[1];
cx node[1],node[6];
cx node[0],node[1];
rz(pi/16) node[0];
cx node[1],node[6];
cx node[2],node[1];
rz(pi/8) node[6];
cx node[1],node[6];
cx node[2],node[1];
cx node[1],node[6];
rz(5.8904862) node[2];
cx node[2],node[1];
cx node[1],node[6];
cx node[2],node[1];
cx node[1],node[6];
rz(pi/8) node[2];
rz(pi/4) node[6];
cx node[6],node[1];
rz(7*pi/4) node[1];
cx node[6],node[1];
rz(pi/4) node[1];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
rz(pi/2) node[6];
rz(-pi/2) node[6];
h node[6];
rz(-pi/2) node[6];
rz(pi/2) node[6];
cx node[5],node[6];
cx node[6],node[5];
cx node[5],node[6];
rz(pi/32) node[7];
cx node[6],node[7];
rz(6.2340979) node[7];
cx node[6],node[7];
rz(pi/32) node[6];
cx node[6],node[1];
rz(6.1850105) node[1];
cx node[6],node[1];
rz(pi/32) node[1];
rz(pi/16) node[6];
cx node[6],node[1];
cx node[1],node[2];
cx node[6],node[1];
cx node[1],node[2];
rz(6.0868358) node[2];
rz(pi/64) node[7];
cx node[2],node[7];
cx node[7],node[2];
cx node[2],node[7];
cx node[6],node[7];
rz(pi/8) node[6];
rz(pi/16) node[7];
rz(pi/512) node[8];
cx node[8],node[3];
rz(6.2770494) node[3];
cx node[8],node[3];
rz(pi/512) node[3];
cx node[3],node[4];
cx node[4],node[3];
cx node[3],node[4];
cx node[3],node[2];
rz(6.2586416) node[2];
cx node[3],node[2];
rz(pi/128) node[2];
rz(pi/64) node[3];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[2],node[1];
rz(6.2340979) node[1];
cx node[2],node[1];
rz(pi/64) node[1];
cx node[1],node[6];
rz(pi/32) node[2];
cx node[2],node[7];
cx node[6],node[1];
cx node[1],node[6];
cx node[1],node[0];
rz(5.8904862) node[0];
cx node[1],node[0];
rz(pi/8) node[0];
rz(pi/4) node[1];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[0],node[5];
rz(7*pi/4) node[5];
cx node[0],node[5];
rz(pi/2) node[0];
rz(-pi/2) node[0];
h node[0];
rz(-pi/2) node[0];
rz(pi/2) node[0];
rz(pi/4) node[5];
rz(6.1850105) node[7];
cx node[2],node[7];
rz(pi/16) node[2];
cx node[2],node[1];
rz(6.0868358) node[1];
cx node[2],node[1];
rz(pi/16) node[1];
rz(pi/8) node[2];
cx node[1],node[2];
cx node[2],node[1];
cx node[1],node[2];
cx node[1],node[0];
cx node[0],node[5];
cx node[1],node[0];
cx node[0],node[5];
rz(5.8904862) node[5];
rz(pi/32) node[7];
cx node[6],node[7];
cx node[7],node[6];
cx node[6],node[7];
rz(pi/256) node[8];
cx node[8],node[3];
rz(6.2709135) node[3];
cx node[8],node[3];
rz(pi/256) node[3];
rz(pi/128) node[8];
cx node[8],node[7];
rz(6.2586416) node[7];
cx node[8],node[7];
rz(pi/128) node[7];
cx node[6],node[7];
rz(pi/64) node[8];
cx node[7],node[8];
cx node[6],node[7];
rz(6.2340979) node[6];
cx node[7],node[8];
cx node[6],node[7];
cx node[7],node[8];
cx node[6],node[7];
rz(pi/64) node[6];
cx node[5],node[6];
cx node[6],node[5];
cx node[5],node[6];
cx node[1],node[6];
rz(pi/4) node[1];
cx node[1],node[0];
rz(7*pi/4) node[0];
cx node[1],node[0];
rz(pi/4) node[0];
rz(pi/2) node[1];
rz(-pi/2) node[1];
h node[1];
rz(-pi/2) node[1];
rz(pi/2) node[1];
rz(pi/8) node[6];
cx node[7],node[8];
rz(pi/32) node[8];
rz(pi/1024) node[9];
cx node[9],node[4];
rz(6.2801173) node[4];
cx node[9],node[4];
rz(pi/1024) node[4];
rz(pi/512) node[9];
cx node[9],node[4];
cx node[4],node[3];
cx node[9],node[4];
cx node[4],node[3];
rz(6.2770494) node[3];
cx node[3],node[8];
cx node[8],node[3];
cx node[3],node[8];
cx node[3],node[2];
rz(6.1850105) node[2];
cx node[3],node[2];
rz(pi/32) node[2];
rz(pi/16) node[3];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[2],node[1];
cx node[1],node[6];
cx node[2],node[1];
cx node[1],node[6];
cx node[0],node[1];
rz(6.0868358) node[6];
cx node[9],node[8];
rz(pi/512) node[8];
rz(pi/256) node[9];
cx node[8],node[9];
cx node[9],node[8];
cx node[8],node[9];
cx node[8],node[7];
rz(6.2709135) node[7];
cx node[8],node[7];
rz(pi/256) node[7];
rz(pi/128) node[8];
cx node[7],node[8];
cx node[8],node[7];
cx node[7],node[8];
cx node[3],node[8];
cx node[6],node[7];
cx node[7],node[6];
cx node[6],node[7];
cx node[2],node[7];
rz(pi/8) node[2];
cx node[1],node[2];
cx node[0],node[1];
rz(5.8904862) node[0];
cx node[1],node[2];
cx node[0],node[1];
cx node[1],node[2];
cx node[0],node[1];
rz(pi/8) node[0];
cx node[1],node[2];
rz(pi/4) node[2];
cx node[2],node[1];
rz(7*pi/4) node[1];
cx node[2],node[1];
rz(pi/4) node[1];
cx node[0],node[1];
rz(pi/2) node[2];
rz(-pi/2) node[2];
h node[2];
rz(-pi/2) node[2];
rz(pi/2) node[2];
cx node[6],node[5];
rz(6.2586416) node[5];
cx node[6],node[5];
rz(pi/128) node[5];
rz(pi/64) node[6];
rz(pi/16) node[7];
cx node[8],node[3];
cx node[3],node[8];
cx node[8],node[7];
cx node[7],node[6];
cx node[8],node[7];
cx node[7],node[6];
rz(6.2340979) node[8];
cx node[8],node[7];
cx node[7],node[6];
cx node[8],node[7];
cx node[7],node[6];
rz(pi/32) node[6];
cx node[6],node[7];
rz(6.1850105) node[7];
cx node[6],node[7];
rz(pi/16) node[6];
cx node[1],node[6];
cx node[0],node[1];
rz(6.0868358) node[0];
cx node[1],node[6];
cx node[0],node[1];
cx node[1],node[6];
cx node[0],node[1];
rz(pi/16) node[0];
cx node[1],node[6];
rz(pi/8) node[6];
cx node[6],node[1];
rz(5.8904862) node[1];
cx node[6],node[1];
rz(pi/8) node[1];
rz(pi/4) node[6];
cx node[1],node[6];
cx node[6],node[1];
cx node[1],node[6];
cx node[1],node[2];
rz(7*pi/4) node[2];
cx node[1],node[2];
rz(pi/2) node[1];
rz(-pi/2) node[1];
h node[1];
rz(-pi/2) node[1];
rz(pi/2) node[1];
rz(pi/4) node[2];
rz(pi/32) node[7];
rz(pi/64) node[8];
