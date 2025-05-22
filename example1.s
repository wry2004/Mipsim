.text
main:
ADDIU $r8, $r0, DATA
ADDIU $r1, $r0, 5       # $r1 = 5
ADDIU $r2, $r0, 10      # $r2 = 10
SW    $r3, 0($r4)       # 存入内存，不依赖前面结果
LW    $r5, 4($r6)       # 从其他地址读，和之前无依赖

.data
.align 2
DATA:
.word 128
BUFFER:
.word 300
