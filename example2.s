.text
main:
ADDIU $r8, $r0, DATA
LW    $r1, 0($r0)       # 从内存加载值到 $r1
ADD   $r2, $r1, $r1     # RAW 冲突：使用了 $r1，依赖上条指令
ADD   $r3, $r2, $r1     # RAW 冲突：使用了 $r2，依赖上条指令
SW    $r3, 0($r4)       # 写回内存

.data
.align 2
DATA:
.word 128
BUFFER:
.word 300
