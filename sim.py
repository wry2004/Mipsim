import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, font
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from copy import deepcopy
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# ---------------- Assembly Parser ----------------
def parse_asm(code_text):
    code_text = code_text.lstrip('\ufeff')
    raw_lines = code_text.splitlines()
    lines = []
    for L in raw_lines:
        m = re.match(r'^\s*([0-9A-Fa-f]+):\s*(.*)$', L)
        lines.append(m.group(2) if m else L)
    labels = {}
    addr = 0
    for L in lines:
        s = L.split('#', 1)[0].strip()
        if not s or s.startswith('#') or s.startswith('.'):
            continue
        m2 = re.match(r'^[A-Za-z_]\w*:$', s)
        if m2:
            labels[m2.group(0)[:-1]] = addr
        else:
            addr += 4
    instrs = []
    for lbl, a in labels.items():
        instrs.append({
            'addr': a,
            'op':   'label',
            'args': [],
            'raw':  lbl
        })
    addr = 0
    for L in lines:
        s = L.split('#', 1)[0].strip()
        if not s or s.startswith('#') or s.startswith('.'):
            continue
        if re.match(r'^[A-Za-z_]\w*:$', s):
            continue
        parts = [p for p in re.split(r'[,\s()]+', s) if p]
        op = parts[0].lower()
        if op == 'beqz' and len(parts) == 3:
            parts.insert(2, '$r0')  # 在 rs 和 label 之间加入 $r0
        args = parts[1:]
        instr = {'addr': addr, 'op': op, 'args': args, 'raw': s}
        if op in ('beq', 'bne', 'beqz', 'bgtz', 'blez', 'bgez', 'bgezal', 'bltz'):
            tgt = args[-1]
            if tgt in labels:
                instr['imm'] = (labels[tgt] - (addr + 4)) // 4
        if op in ('addi', 'addiu'):
            tgt = args[2]
            instr['imm'] = labels[tgt] if tgt in labels else int(tgt, 0)
        instrs.append(instr)
        addr += 4
    instrs.sort(key=lambda ins: ins['addr'])
    return instrs

# ---------------- Pipeline CPU ----------------
class PipelineCPU:
    def __init__(self, instr_list):
        self.instr_list = instr_list[:]
        self.instrs = {i['addr']: i for i in instr_list if i['op'] != 'label'}
        self.PC = 0
        self.regs = [0] * 32
        self.mem = [0] * 4096
        self.pipeline = {'IF': None, 'ID': None, 'EX': None, 'MEM': None, 'WB': None}
        self.stats = {
            'cycles': 0,
            'id_count': 0,            # ID段执行指令数
            'branch_count': 0,        # 分支指令条数
            'load_count': 0,          # load 指令数
            'store_count': 0,         # store 指令数
            'adder_cycles': 0,        # 加法器使用周期
            'mul_cycles': 0,          # 乘法器使用周期
            'div_cycles': 0           # 除法器使用周期
        }
        self.halted = False
        self.timeline = []
        self.hazards = []
        self.history = []
        self.break_addr = None
    def step(self):
        self.history.append((self.PC, list(self.regs), list(self.mem),
                             {k: (v.copy() if isinstance(v, dict) else v) for k, v in self.pipeline.items()},
                             dict(self.stats), self.halted, self.break_addr))
        if self.pipeline['ID']:
            self.stats['id_count'] += 1
        wb = self.pipeline['WB']
        if wb:
            if wb['op'] == 'lw':
                self.regs[int(wb['args'][0][2:])] = wb['mem_val']
            elif wb['op'] in ('add', 'addi', 'addiu'):
                self.regs[int(wb['args'][0][2:])] = wb['exe_val']
        self.pipeline['WB'] = self.pipeline['MEM']
        self.pipeline['MEM'] = self.pipeline['EX']
        id_inst = self.pipeline['ID']
        hazard = False
        if id_inst:
            srcs = []
            op,a = id_inst['op'], id_inst['args']
            if op in ('add','addi','addiu'):
                srcs.append(int(a[1][2:]))
                if op=='add': srcs.append(int(a[2][2:]))
            elif op in ('lw','sw'):
                srcs.append(int(a[2][2:]))
                if op=='sw': srcs.append(int(a[0][2:]))
            elif op in ('beqz','beq','bne'):
                srcs.append(int(a[0][2:]))
                srcs.append(int(a[1][2:]))
            for stage in ('EX','MEM','WB'):
                inst = self.pipeline[stage]
                if inst and inst['op'] in ('add','addi','addiu','lw'):
                    dst = int(inst['args'][0][2:])
                    if dst in srcs:
                        hazard = True
                        break
        if hazard:
            self.pipeline['EX'] = None
            mem_node = self.pipeline['MEM']
            if mem_node and mem_node['op'] == 'lw':
                mem_node['mem_val'] = self.mem[mem_node['mem_addr']]
            elif mem_node and mem_node['op'] == 'sw':
                self.mem[mem_node['mem_addr']] = self.regs[int(mem_node['args'][0][2:])]
            self.stats['cycles'] += 1
            self.timeline.append(self.pipeline.copy())
            self.hazards.append([( 'ID', None, None, id_inst )])
            return
        ex = deepcopy(self.pipeline['ID'])
        ex_copy = deepcopy(ex)
        branch_taken = False
        if ex:
            op, a = ex['op'], ex['args']
            if op == 'add':
                self.stats['adder_cycles'] += 1
                ex['exe_val'] = self.regs[int(a[1][2:])] + self.regs[int(a[2][2:])]
            elif op in ('addi', 'addiu'):
                self.stats['adder_cycles'] += 1
                imm = ex.get('imm', 0)
                ex['exe_val'] = self.regs[int(a[1][2:])] + imm
            elif op in ('lw', 'sw'):
                self.stats['load_count'] += 1
                base = int(a[2][2:]); off = int(a[1], 0)
                ex['mem_addr'] = self.regs[base] + off
            elif op == 'beqz':
                self.stats['branch_count'] += 1
                if self.regs[int(a[0][2:])] == 0:
                    self.PC = ex_copy['addr'] + 4 * (ex['imm'] + 1)
                    branch_taken = True
            elif op == 'beq':
                self.stats['branch_count'] += 1
                if self.regs[int(a[0][2:])] == self.regs[int(a[1][2:])]:
                    self.PC = ex_copy['addr'] + 4 * (ex['imm'] + 1)
                    branch_taken = True
            elif op == 'bne':
                self.stats['branch_count'] += 1
                if self.regs[int(a[0][2:])] != self.regs[int(a[1][2:])]:
                    self.PC = ex_copy['addr'] + 4 * (ex['imm'] + 1)
                    branch_taken = True
            elif op == 'j':
                self.stats['branch_count'] += 1
                self.PC = ex_copy['target_addr']
                branch_taken = True
            elif op == 'jal':
                self.stats['branch_count'] += 1
                self.regs[31] = ex_copy['addr'] + 4
                self.PC = ex_copy['target_addr']
                branch_taken = True
            elif op == 'jr':
                self.stats['branch_count'] += 1
                self.PC = self.regs[int(a[0][2:])]
                branch_taken = True
        self.pipeline['EX'] = ex
        mem_node = self.pipeline['MEM']
        if mem_node and mem_node['op'] == 'lw':
            mem_node['mem_val'] = self.mem[mem_node['mem_addr']]
        elif mem_node and mem_node['op'] == 'sw':
            self.mem[mem_node['mem_addr']] = self.regs[int(mem_node['args'][0][2:])]
        self.pipeline['ID'] = self.pipeline['IF']
        mem_stall = self.pipeline['MEM'] and self.pipeline['MEM']['op'] in ('lw', 'sw')
        id_inst = self.pipeline['ID']
        branch_pending = id_inst and id_inst['op'] in ('beqz','beq','bne','bgtz','blez','bgez','bgezal','bltz','j','jal','jr')
        if not mem_stall and not branch_pending and not branch_taken:
            self.pipeline['IF'] = deepcopy(self.instrs.get(self.PC))
            self.PC += 4
        else:
            self.pipeline['IF'] = None
        if self.break_addr is not None and self.PC == self.break_addr:
            self.halted = True
        self.stats['cycles'] += 1
        if all(stage is None for stage in self.pipeline.values()):
            self.halted = True
        self.timeline.append(self.pipeline.copy())
        haz = []
        id_inst = self.pipeline['ID']
        if id_inst:
            srcs = []
            op, a = id_inst['op'], id_inst['args']
            if op in ('add', 'addi', 'addiu'):
                srcs.append(int(a[1][2:]))
                if op == 'add': srcs.append(int(a[2][2:]))
            elif op in ('lw', 'sw'):
                srcs.append(int(a[2][2:]))
                if op == 'sw': srcs.append(int(a[0][2:]))
            elif op in ('beqz','beq','bne'):
                srcs.append(int(a[0][2:]))
                srcs.append(int(a[1][2:]))
            for stage in ('EX', 'MEM', 'WB'):
                inst = self.pipeline[stage]
                if inst and inst['op'] in ('add', 'addi', 'addiu', 'lw'):
                    dst = int(inst['args'][0][2:])
                    if dst in srcs:
                        haz.append((stage, inst, dst, id_inst))
        self.hazards.append(haz)
    def rollback(self):
        if not self.history:
            return
        self.PC, regs, mem, pipe, stats, halted, bp = self.history.pop()
        self.regs, self.mem = regs, mem
        self.pipeline, self.stats = pipe, stats
        self.halted, self.break_addr = halted, bp
        if self.timeline: self.timeline.pop()
        if self.hazards: self.hazards.pop()

# ---------------- Pipeline CPU with Forwarding (Fixed) ----------------
class PipelineCPU2:
    def __init__(self, instr_list):
        self.instr_list = instr_list[:]
        self.instrs = {i['addr']: i for i in instr_list if i['op'] != 'label'}
        self.PC = 0
        self.regs = [0] * 32
        self.mem = [0] * 4096
        self.pipeline = {'IF': None, 'ID': None, 'EX': None, 'MEM': None, 'WB': None}
        self.stats = {'cycles':0, 'id_count':0, 'branch_count':0, 'load_count':0,
                      'store_count':0, 'adder_cycles':0, 'mul_cycles':0, 'div_cycles':0}
        self.halted = False
        self.timeline = []
        self.hazards = []
        self.history = []
        self.break_addr = None
    def step(self):
        self.history.append((self.PC, list(self.regs), list(self.mem), deepcopy(self.pipeline), dict(self.stats), self.halted, self.break_addr))
        if self.pipeline['ID']:
            self.stats['id_count'] += 1
        wb = self.pipeline['WB']
        if wb:
            if wb['op']=='lw' and 'mem_val' not in wb:
                wb['mem_val'] = self.mem[wb['mem_addr']]
            if wb['op']=='lw':
                self.regs[int(wb['args'][0][2:])] = wb['mem_val']
            elif wb['op'] in ('add','addi','addiu'):
                self.regs[int(wb['args'][0][2:])] = wb['exe_val']
        self.pipeline['WB'] = self.pipeline['MEM']
        self.pipeline['MEM'] = self.pipeline['EX']
        id_node = self.pipeline['ID']
        ex = deepcopy(id_node)
        branch_taken = False
        if ex:
            op,a = ex['op'],ex['args']
            def forward(reg_idx):
                memn = self.pipeline['MEM']
                if memn and memn['args'] and memn['op'] in ('add','addi','addiu','lw'):
                    dst = int(memn['args'][0][2:])
                    if dst == reg_idx:
                        if memn['op']=='lw':
                            if 'mem_val' not in memn:
                                memn['mem_val'] = self.mem[memn['mem_addr']]
                            return memn['mem_val']
                        if 'exe_val' in memn:
                            return memn['exe_val']
                wbn = self.pipeline['WB']
                if wbn and wbn['args'] and wbn['op'] in ('add','addi','addiu','lw'):
                    dst = int(wbn['args'][0][2:])
                    if dst == reg_idx:
                        if wbn['op']=='lw':
                            if 'mem_val' not in wbn:
                                wbn['mem_val'] = self.mem[wbn['mem_addr']]
                            return wbn['mem_val']
                        if 'exe_val' in wbn:
                            return wbn['exe_val']
                return self.regs[reg_idx]
            if op=='add':
                self.stats['adder_cycles']+=1
                v1=forward(int(a[1][2:])); v2=forward(int(a[2][2:])); ex['exe_val']=v1+v2
            elif op in ('addi','addiu'):
                self.stats['adder_cycles']+=1
                v1=forward(int(a[1][2:])); ex['exe_val']=v1+ex.get('imm',0)
            elif op=='lw':
                self.stats['load_count']+=1
                base=forward(int(a[2][2:])); off=int(a[1],0); ex['mem_addr']=base+off
            elif op=='sw':
                self.stats['store_count']+=1
                base=forward(int(a[2][2:])); off=int(a[1],0)
                ex['mem_addr']=base+off; ex['store_val']=forward(int(a[0][2:]))
            elif op in ('beq','bne','beqz'):
                self.stats['branch_count']+=1
                v1=forward(int(a[0][2:])); v2=forward(int(a[1][2:]))
                take = (op=='beq' and v1==v2) or (op=='bne' and v1!=v2) or (op=='beqz' and v1==0)
                if take:
                    self.PC = ex['addr']+4*(ex['imm']+1)
                    branch_taken=True
        self.pipeline['EX'] = ex
        memn = self.pipeline['MEM']
        if memn:
            if memn['op']=='lw': memn['mem_val'] = self.mem[memn['mem_addr']]
            elif memn['op']=='sw': self.mem[memn['mem_addr']] = memn['store_val']
        self.pipeline['ID'] = self.pipeline['IF']
        mem_stall = False
        exn = self.pipeline['EX']
        idn = self.pipeline['ID']
        if exn and exn['op']=='lw' and idn:
            rt = int(exn['args'][0][2:])
            srcs = []
            if idn['op'] in ('add','addi','addiu','lw','sw'):
                for i in (1,2):
                    if i < len(idn['args']): srcs.append(int(idn['args'][i][2:]))
            if rt in srcs:
                mem_stall = True
        branch_pending = idn and idn['op'] in ('beq','bne','beqz','j','jal','jr')
        if not mem_stall and not branch_pending and not branch_taken:
            self.pipeline['IF'] = deepcopy(self.instrs.get(self.PC)); self.PC += 4
        else:
            self.pipeline['IF'] = None
        if self.break_addr is not None and self.PC == self.break_addr:
            self.halted = True
        self.stats['cycles'] += 1
        if all(stage is None for stage in self.pipeline.values()):
            self.halted = True
        self.stats['cycles'] += 1
        self.timeline.append(deepcopy(self.pipeline))
        self.hazards.append([])
        if all(stage is None for stage in self.pipeline.values()):
            self.halted = True
    def rollback(self):
        if not self.history: return
        self.PC,regs,mem,pipe,stats,halted,bp = self.history.pop()
        self.regs, self.mem = regs, mem
        self.pipeline, self.stats = pipe, stats
        self.halted, self.break_addr = halted, bp
        if self.timeline: self.timeline.pop()
        if self.hazards: self.hazards.pop()

# ---------------- GUI ----------------
class App:
    def __init__(self, root):
        self.cpu = None
        self.forwarding_enabled = True
        self.instrs = None
        self.bps = set()
        root.title("MIPS 5 段流水线模拟器")
        root.state('zoomed')
        self.base_font_size = 12
        self.base_font = font.Font(family='Microsoft YaHei', size=self.base_font_size)
        for i in range(2): root.grid_rowconfigure(i, weight=1)
        for j in range(3): root.grid_columnconfigure(j, weight=1)
        titles = [('统计', 0, 0), ('代码', 0, 1), ('寄存器', 0, 2),
                  ('流水线', 1, 0), ('内存', 1, 1), ('时钟周期图', 1, 2)]
        frames = {}
        for t, r, c in titles:
            f = tk.LabelFrame(root, text=t, labelanchor='n', font=self.base_font)
            f.grid(row=r, column=c, sticky='nsew', padx=2, pady=2)
            frames[t] = f
        self.txt_stats = scrolledtext.ScrolledText(frames['统计'], font=self.base_font)
        self.txt_stats.pack(fill=tk.BOTH, expand=True)
        self.txt_code  = scrolledtext.ScrolledText(frames['代码'], font=self.base_font)
        self.txt_code.pack(fill=tk.BOTH, expand=True)
        self.txt_code.tag_configure('bp', background='lightyellow')
        self.txt_reg   = scrolledtext.ScrolledText(frames['寄存器'], font=self.base_font)
        self.txt_reg.pack(fill=tk.BOTH, expand=True)
        self.txt_pipe  = scrolledtext.ScrolledText(frames['流水线'], font=self.base_font)
        self.txt_pipe.pack(fill=tk.BOTH, expand=True)
        self.txt_mem   = scrolledtext.ScrolledText(frames['内存'], font=self.base_font)
        self.txt_mem.pack(fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, frames['时钟周期图'])
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ctrl = tk.Frame(root)
        ctrl.grid(row=2, column=0, columnspan=3, sticky='ew')
        tk.Label(ctrl, text="字体大小:", font=self.base_font).pack(side=tk.LEFT, padx=(10,0))
        self.font_scale = tk.Scale(ctrl, from_=8, to=24, orient=tk.HORIZONTAL,
                                   command=self.on_font_change, font=self.base_font)
        self.font_scale.set(self.base_font_size)
        self.font_scale.pack(side=tk.LEFT)
        for text, cmd in [("加载 ASM/.s", self.load), ("单步", self.step),
                          ("回退", self.rollback), ("运行结束", self.run)]:
            tk.Button(ctrl, text=text, command=cmd, font=self.base_font).pack(side=tk.LEFT)
        tk.Label(ctrl, text="断点:", font=self.base_font).pack(side=tk.LEFT)
        self.bp_entry = tk.Entry(ctrl, width=8, font=self.base_font)
        self.bp_entry.pack(side=tk.LEFT)
        tk.Button(ctrl, text="设置断点", command=self.set_bp, font=self.base_font).pack(side=tk.LEFT)
        tk.Button(ctrl, text="重置状态", command=self.reset, font=self.base_font).pack(side=tk.LEFT, padx=10)
        tk.Button(ctrl, text="清空全部", command=self.clear_all, font=self.base_font).pack(side=tk.LEFT)
        self.fwd_var=tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="定向路径", variable=self.fwd_var,
                       command=self.toggle_forwarding, font=self.base_font).pack(side=tk.LEFT)
    def on_font_change(self, size):
        self.base_font_size = int(size)
        self.base_font.configure(size=self.base_font_size)
        for widget in [self.txt_stats, self.txt_code, self.txt_reg, self.txt_pipe, self.txt_mem]:
            widget.configure(font=self.base_font)
        self.ax.tick_params(labelsize=self.base_font_size)
        self.ax.xaxis.label.set_size(self.base_font_size)
        self.ax.yaxis.label.set_size(self.base_font_size)
        self.canvas.draw()
    def toggle_forwarding(self):
        self.forwarding_enabled=self.fwd_var.get(); self.refresh()
    def load(self):
        path = filedialog.askopenfilename(filetypes=[('ASM/S','*.asm *.s')])
        if not path: return
        txt = open(path, encoding='utf-8', errors='ignore').read()
        self.instrs = parse_asm(txt)
        if self.forwarding_enabled:
            self.cpu = PipelineCPU2(self.instrs)
        else:
            self.cpu = PipelineCPU(self.instrs)
        self.bps.clear()
        self.refresh()
    def refresh(self):
        if not self.cpu: return
        self.txt_code.delete('1.0', tk.END)
        for instr in self.cpu.instr_list:
            tag = 'bp' if instr['addr'] in self.bps else None
            if instr['op'] == 'label':
                self.txt_code.insert(tk.END, f"{instr['raw']}\n", tag)
            else:
                self.txt_code.insert(tk.END, f"{instr['addr']:04x}: {instr['raw']}\n", tag)
        self.txt_reg.delete('1.0', tk.END)
        regs = self.cpu.regs
        cols = 4
        rows = (len(regs) + cols - 1) // cols
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx < len(regs):
                    self.txt_reg.insert(tk.END, f"R{idx:02d}={regs[idx]:<6}")
                    if c < cols - 1:
                        self.txt_reg.insert(tk.END, "   \t ")
            self.txt_reg.insert(tk.END, '\n\n')
        self.txt_pipe.delete('1.0', tk.END)
        last_haz = self.cpu.hazards[-1] if self.cpu.hazards else []
        haz_map = {}
        for stage, prod, reg, cons in last_haz:
            if prod is None or cons is None:
                continue
            haz_map.setdefault(stage, []).append((prod, reg, cons))
        stages = ['IF', 'ID', 'EX', 'MEM', 'WB']
        for s in stages:
            inst = self.cpu.pipeline[s]
            self.txt_pipe.insert(tk.END, f"{s}: {inst['raw'] if inst else '---'}\n")
            if s in haz_map:
                for prod, reg, cons in haz_map[s]:
                    info = f"发生RAW冲突:\n指令 “ {cons['raw']} ”\n依赖来自指令\n“ {prod['raw']} ”\n的 R{reg} "
                    self.txt_pipe.insert(tk.END, info + "\n")
            self.txt_pipe.insert(tk.END, "\n" * 3)
        self.txt_pipe.tag_configure('haz', background='lightpink')
        for s in haz_map:
            idx = self.txt_pipe.search(f"^{s}:", '1.0', tk.END, regexp=True)
            if idx:
                self.txt_pipe.tag_add('haz', idx, f"{idx} lineend")
        self.txt_mem.delete('1.0', tk.END)
        for a in range(0, 64, 4):
            self.txt_mem.insert(tk.END, f"{a:04x}: {self.cpu.mem[a:a+4]}\n")
        stall_count = 0
        raw_count = 0
        for haz in self.cpu.hazards:
            for stage, prod, reg, cons in haz:
                if prod is None:
                    stall_count += 1
                else:
                    raw_count += 1
        fwd_status = "启用" if self.forwarding_enabled else "禁用"
        stats_str = (
            f"定向路径: {fwd_status}\n\n"
            f"执行周期总数周期: {self.cpu.stats['cycles']}   PC=0x{self.cpu.PC:04x}\n\n"
            f"ID段执行指令数: {self.cpu.stats['id_count']}\n\n"
            f"总RAW停顿: {stall_count}\n\n"
            f"分支指令: {self.cpu.stats['branch_count']}\n\n"
            f"Load 指令: {self.cpu.stats['load_count']}\n\n"   
            f"Store 指令: {self.cpu.stats['store_count']}\n\n"
            f"加法器运行时间（周期数）: {self.cpu.stats['adder_cycles']}\n\n"   
            f"乘法器运行时间（周期数）: {self.cpu.stats['mul_cycles']}\n\n"   
            f"除法器运行时间（周期数）: {self.cpu.stats['div_cycles']}\n\n"
        )
        self.txt_stats.delete('1.0', tk.END)
        self.txt_stats.insert('1.0', stats_str)
        instrs = sorted(self.cpu.instrs.values(), key=lambda x: x['addr'])
        addrs = [instr['addr'] for instr in instrs]
        cycles = len(self.cpu.timeline)
        stage_matrix = np.zeros((cycles, len(addrs)), dtype=int)
        idx_map = {'IF':1,'ID':2,'EX':3,'MEM':4,'WB':5,'STALL':6}
        for cyc, pipe in enumerate(self.cpu.timeline):
            for col, instr in enumerate(instrs):
                for stg, idx in idx_map.items():
                    if stg.startswith('STALL'): continue
                    node = pipe.get(stg)
                    if node and node['addr']==instr['addr']:
                        stage_matrix[cyc, col] = idx
                        break
                prev_id = self.cpu.timeline[cyc-1]['ID'] if cyc>0 else None
                cur_id  = pipe['ID']
                if prev_id and cur_id and cur_id['addr']==prev_id['addr']==instr['addr']:
                    stage_matrix[cyc, col] = idx_map['STALL']
                prev_if = self.cpu.timeline[cyc-1]['IF'] if cyc>0 else None
                cur_if  = pipe['IF']
                if prev_if and cur_if and cur_if['addr']==prev_if['addr']==instr['addr']:
                    stage_matrix[cyc, col] = idx_map['STALL']
        self.ax.clear()
        cmap = ListedColormap(['white','yellow','green','red','blue','orange','gray'])
        norm = BoundaryNorm(boundaries=list(range(8)), ncolors=cmap.N)
        self.ax.imshow(stage_matrix.T,
                       aspect='auto', interpolation='none',
                       cmap=cmap, norm=norm,
                       origin='lower', extent=[0, cycles, 0, len(addrs)])
        self.ax.set_xlabel('时钟周期')
        self.ax.set_ylabel('指令地址')
        self.ax.set_xticks(np.arange(cycles))
        self.ax.set_xticklabels([str(i) for i in range(cycles)])
        self.ax.set_yticks(range(len(addrs)))
        self.ax.set_yticklabels([f"{addr:04x}" for addr in addrs])
        legend = [Patch(facecolor=cmap(i), label=l) for l,i in [('IF',1),('ID',2),('EX',3),('MEM',4),('WB',5),('STALL',6)]]
        self.ax.legend(handles=legend, bbox_to_anchor=(1.16,1), loc='upper right')
        self.canvas.draw()
    def step(self):
        if not self.cpu: return
        self.cpu.halted = False
        old_bp = self.cpu.break_addr
        self.cpu.break_addr = None
        self.cpu.step()
        self.cpu.break_addr = old_bp
        self.refresh()
    def rollback(self):
        if not self.cpu: return
        self.cpu.rollback()
        self.refresh()
    def run(self):
        if not self.cpu: return
        while not self.cpu.halted:
            self.cpu.step()
        self.refresh()
    def set_bp(self):
        if not self.cpu: return
        try:
            addr = int(self.bp_entry.get().strip(), 16)
        except ValueError:
            messagebox.showerror("错误", "断点格式错误")
            return
        self.bps.add(addr)
        self.cpu.break_addr = addr
        self.refresh()
    def reset(self):
        if not self.instrs: return
        self.cpu = PipelineCPU(self.instrs)
        self.bps.clear()
        self.refresh()
    def clear_all(self):
        self.cpu = None; self.instrs = None; self.bps.clear()
        for w in [self.txt_code, self.txt_reg, self.txt_pipe, self.txt_mem, self.txt_stats]:
            w.delete('1.0', tk.END)
        self.ax.clear(); self.canvas.draw()
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
