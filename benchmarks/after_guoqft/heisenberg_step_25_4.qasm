OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[1];
h q[3];
sdg q[1];
s q[3];
sdg q[1];
s q[3];
h q[1];
h q[3];

