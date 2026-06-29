
参考 [Scriptware Commands — UMR: User Mode Register Debugger documentation](https://umr.readthedocs.io/en/main/scriptware.html#pci-instances)

查看具体的pci通道
```
rocm-smi --showbus

============================ ROCm System Management Interface ============================
======================================= PCI Bus ID =======================================
GPU[0]          : PCI Bus: 0000:0A:00.0
GPU[1]          : PCI Bus: 0000:0A:00.1
GPU[2]          : PCI Bus: 0000:0A:00.2
GPU[3]          : PCI Bus: 0000:0A:00.3
GPU[4]          : PCI Bus: 0000:80:00.0
GPU[5]          : PCI Bus: 0000:80:00.1
GPU[6]          : PCI Bus: 0000:80:00.2
GPU[7]          : PCI Bus: 0000:80:00.3
GPU[8]          : PCI Bus: 0000:A4:00.0
GPU[9]          : PCI Bus: 0000:A4:00.1
GPU[10]         : PCI Bus: 0000:A4:00.2
GPU[11]         : PCI Bus: 0000:A4:00.3
GPU[12]         : PCI Bus: 0000:C8:00.0
GPU[13]         : PCI Bus: 0000:C8:00.1
GPU[14]         : PCI Bus: 0000:C8:00.2
GPU[15]         : PCI Bus: 0000:C8:00.3
GPU[16]         : PCI Bus: 0001:0B:00.0
GPU[17]         : PCI Bus: 0001:0B:00.1
GPU[18]         : PCI Bus: 0001:0B:00.2
GPU[19]         : PCI Bus: 0001:0B:00.3
GPU[20]         : PCI Bus: 0001:81:00.0
GPU[21]         : PCI Bus: 0001:81:00.1
GPU[22]         : PCI Bus: 0001:81:00.2
GPU[23]         : PCI Bus: 0001:81:00.3
GPU[24]         : PCI Bus: 0001:A5:00.0
GPU[25]         : PCI Bus: 0001:A5:00.1
GPU[26]         : PCI Bus: 0001:A5:00.2
GPU[27]         : PCI Bus: 0001:A5:00.3
GPU[28]         : PCI Bus: 0001:C9:00.0
GPU[29]         : PCI Bus: 0001:C9:00.1
GPU[30]         : PCI Bus: 0001:C9:00.2
GPU[31]         : PCI Bus: 0001:C9:00.3
==========================================================================================
================================== End of ROCm SMI Log ===================================
```
查看pci-instance 和 gpu device的对应关系
```
./umr --script instances
-> 41 25 17 0 57 9 49 33

./umr --script pci-bus 17
-> 0000:a4:00.0

```

merge the core with gpucore
![](./Pasted%20image%2020250926230406.png)

### Top level GPU Status Registers:  
GRBM_STATUS  
GRBM_STATUS2  
GRBM_STATUS_SE0  
GRBM_STATUS_SE1  
GRBM_STATUS_SE2  
GRBM_STATUS_SE3

CP block level Status registers:  
CP_STAT  
CP_BUSY  
CP_BUSY_STAT

CP_STALLED_STAT1  
CP_STALLED_STAT2  
CP_STALLED_STAT3

CP_CPC_STATUS  
CP_CPC_BUSY_STAT  
CP_CPC_STALLED_STAT1

CP_CPF_STATUS  
CP_CPF_BUSY_STAT  
CP_CPF_STALLED_STAT1


### cpc result
umr -g mi300@0 -vmp 0 -cpc

Pipe 0  Queue 4  VMID 3
  PQ BASE 0x7f999d094000  RPTR 0x170  WPTR 0x180  RPTR_ADDR 0x7f999d096080  CNTL 0x14014509
  EOP BASE 0x0  RPTR 0x40000000  WPTR 0x7f8000  WPTR_MEM 0x0
  MQD 0x6ed000  DEQ_REQ 0x0  IQ_TIMER 0x2000000  AQL_CONTROL 0x43
  SAVE BASE 0x7f77fd000000  SIZE 0xba6000  STACK OFFSET 0x2000  SIZE 0x2000

 
ME 1 Pipe 0: INSTR_PTR 0x3d25
 

These fields can generally give some idea of CP FW state.
RPTR and WPTR are the read and write index. They are in DWORDs (32-bits). Each AQL packet is 64-bytes so each time 1 packet is inserted into the queue, the WPTR is incremented by 0x10. In the example above, there is 1 packet still in the queue.

Then the AQL_CONTROL bits can also be useful. These are status bits.
In this example,  0x43 = 0100 0011 (binary)
So bits 0, 1 and 6 are set. These are the bit definitions:

//--------------------------------------------------
// Aql defines
//
// SCHSAInterfaceVersion "SCHSAInterface 3/28/2014 (Sys Arch 1.0 Provisional)"
//--------------------------------------------------
// CP_HQD_AQL_CONTROL
//     31 - write enable bits [30:16]
//  30:29 - unused
//  28:24 - Barrier 'AND' signal status
//     22 - unused
//  21:16 - error code to send
//     15 - write enable bits [14:00]
//  14:12 - eop state
//     11 - Use Scratch Once Active - block the queue
//  10:09 - Post IB Release Fence
//     08 - Processing IB
//     07 - cache flushed
//     06 - send eop
//     05 - error siganled
//     
//     03 - queue blocked waiting for EOP to be empty
//     02 - Aql dispatch active
//     01 - barrier retry
//     00 - Aql queue enable
The last item that is also useful is the instruction pointer (INSTR_PTR). If you send the INST_PTR and the firmware version to a CP FW engineer, they can see the current instruction that CP FW is executing, so they will be able to tell the state of CP FW.