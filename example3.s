.text
main:
ADDIU $r8, $r0, DATA
ADDIU $r1, $r0, 0       # $r1 = 0
BEQZ  $r1, label1       # 如果 $r1 为零，则跳转
ADDIU $r2, $r0, 10      # 这行应被跳过

label1:
ADDIU $r3, $r0, 99      # 目的跳转位置
SW    $r3, 0($r4)       # 存储 $r3 到内存

.data
.align 2
DATA:
.word 128
BUFFER:
.word 300
