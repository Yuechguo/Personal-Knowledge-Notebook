setup env flags:
``` shell
export ROCBLAS_LAYER=2
export TENSILE_DB=255
```

for rocblas logging control:
 ref:https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/logging.html

for tensile logging control:
ref:https://github.com/ROCm/Tensile/wiki/Environment-Variables

running benchmark and get print gemm info:
``` shell
./rocblas-bench -f gemm -r f16_r --transposeA N --transposeB N -m 2 -n 166 -k 64 --alpha 1 --lda 2 --ldb 64 --beta 0 --ldc 2
TensorDescriptor:calculate  3-tensor<Half>( sizes(2, 64, 1), strides(1, 2, 0), offset(0))totalLogicalElements=128 totalAllocatedElem=128
TensorDescriptor:calculate  3-tensor<Half>( sizes(64, 166, 1), strides(1, 64, 0), offset(0))totalLogicalElements=10624 totalAllocatedElem=10624
TensorDescriptor:calculate  3-tensor<Half>( sizes(2, 166, 1), strides(1, 2, 0), offset(0))totalLogicalElements=332 totalAllocatedElem=332
TensorDescriptor:calculate  3-tensor<Half>( sizes(2, 166, 1), strides(1, 2, 0), offset(0))totalLogicalElements=332 totalAllocatedElem=332
TruePred: 1And(): 1
TruePred: 1Kernel Cijk_Ailk_Bljk_HB_MT32x16x32_SN_1LDSB0_APM1_ABV0_ACED0_AF0EM2_AF1EM1_AMAS3_ASE_ASGT_ASLT_ASM_ASAE01_ASCE01_ASEM2_AAC0_BL1_BS1_CLR0_DTLA0_DTLB0_DTVA0_DTVB0_DVO0_ETSP_EPS1_ELFLR0_EMLL0_FSSC10_FL0_GLVWA2_GLVWB2_GRCGA1_GRCGB1_GRPM1_GRVW2_GSU1_GSUASB_GLS0_ISA942_IU1_K1_KLA_LBSPPA0_LBSPPB0_LPA0_LPB0_LDL1_LRVW2_LWPMn1_LDW0_FMA_MIAV0_MDA2_MO40_MMFSC_MKFGSU256_NTA0_NTB0_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_SIA1_SS0_SU32_SUM0_SUS256_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TSGRA0_TSGRB0_TT2_2_TLDS0_UMLDSA0_UMLDSB0_U64SL1_USFGRO1_VAW2_VS1_VW2_VWB2_VFLRP0_WSGRA0_WSGRB0_WS64_WG16_8_1_WGM1
 l(128, 1, 1) x g(1, 11, 1) = (128, 11, 1)
[0..7] tensor2dSizeA: 80 00 00 00 00 00 00 00 (128)
[8..15] tensor2dSizeB: 80 29 00 00 00 00 00 00 (10624)
[16..23] d: 00 2b b0 cb 60 7f 00 00 (0x7f60cbb02b00)
[24..31] c: 00 2b b0 cb 60 7f 00 00 (0x7f60cbb02b00)
[32..39] a: 00 aa 2f d8 60 7f 00 00 (0x7f60d82faa00)
[40..47] b: 00 7e b0 cb 60 7f 00 00 (0x7f60cbb07e00)
[48..49] alpha: 00 3c (1)
[50..51] alpha_2: 00 3c (1)
[52..53] beta: 00 00 (0)
[54..55] beta_2: 00 00 (0)
[56..59] strideD1: 02 00 00 00 (2)
[60..63] strideD2: 00 00 00 00 (0)
[64..67] strideC1: 02 00 00 00 (2)
[68..71] strideC2: 00 00 00 00 (0)
[72..75] strideA1: 02 00 00 00 (2)
[76..79] strideA2: 00 00 00 00 (0)
[80..83] strideB1: 40 00 00 00 (64)
[84..87] strideB2: 00 00 00 00 (0)
[88..91] size_0: 02 00 00 00 (2)
[92..95] size_1: a6 00 00 00 (166)
[96..99] size_2: 01 00 00 00 (1)
[100..103] size_3: 40 00 00 00 (64)
[104..107] staggerUIter: 00 00 00 00 (0)
[108..111] problemNumGroupTiles0: 01 00 00 00 (1)
[112..115] problemNumGroupTiles1: 0b 00 00 00 (11)
[116..119] pad: 00 00 00 00 (0)
Occupancy = 9
```

In the case above, the gemm layout is $M=2,N=166,K=64$ , searching and extracting all the keywords(m, n ,k) from the logs can help to get all the GEMM layouts used in the model.