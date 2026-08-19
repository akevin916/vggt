# 縮寫  gt=GT  mt=monst3r(=m3)  vt=vggt-base(VGGT-1B)  v*=vggt-inst(v3 s1_inst)
# 每方法記三樣: 背景 / 動態物體 / 尺度·drift·floaters
# 註: base↔inst 都對齊到同一 GT frame(相機中心 Sim3),但尺度差大的場景(ambush_5/6、cave_2、
#     temple_3、market_2)兩者本來就不會疊 —— 那是真差異,不是對齊壞掉。

cave_2:
    gt: 背景混雜只有照到的地方有3D點中間背景有很多中空的地方
    m3: 背景混亂不統一，快速轉換的視角沒辦法統一場景，動態前景的怪物重建的不完整

temple_2:
    gt: 背景統一且明顯
    mt: 重構失敗整個場景混雜

sleeping_2:
    gt: 背景統一且明顯
    mt: 品質一般缺少人物的重建
    vt: 建立完整接近gt

alley_2:
    gt: 可辨識
    mt:
    vt: 建立完整接近gt
    v*:人物重建失敗背景牆邊以外的失敗

ambush_4:
    gt: 可辨識
    vt:
    v*:場景整合跟一致性好前後景分離

ambush_5:
    gt:
    mt:
    vt:
    v*:            # 尺度: inst 比 base 大 2.5x, 兩者不疊(真差異)

ambush_6:
    gt:
    mt:
    vt:
    v*:            # 尺度: inst 只有 base 的 0.22x, 差很多

cave_4:
    gt:
    mt:
    vt:
    v*:

market_2:
    gt:
    mt:
    vt:
    v*:            # f50 質心差 26, 檢查是否 drift

market_5:
    gt:
    mt:
    vt:
    v*:

market_6:
    gt:
    mt:
    vt:
    v*:

shaman_3:
    gt:
    mt:
    vt:
    v*:            # base/inst 疊得最好的一組(可當對照)

sleeping_1:
    gt:
    mt:
    vt:
    v*:

temple_3:
    gt:
    mt:
    vt:
    v*:            # 尺度爆: inst scale=42.9, 有效點驟減


場景	base	inst	幾何較好
alley_2	0.270	0.176	inst
ambush_4	0.100	0.226	base
ambush_5	0.011	0.012	≈
ambush_6	0.084	0.040	inst
cave_2	0.341	0.484	base(都爛)
cave_4	0.051	0.041	inst
market_2	⚠️0.349	0.855	都失敗
market_5	0.211	0.139	inst
market_6	0.211	0.167	inst
shaman_3	0.0072	0.0055	inst
sleeping_1	0.022	0.019	inst
sleeping_2	0.016	0.018	≈
temple_2	0.418	0.852	base(都爛)
temple_3	0.187	0.511	base



## waymo
### 1100
mt: 很好
vggt: 天空分不開

### 139654609945
mt: 天空融到樹上
base: 比mt好一點
smooth: 比base好一點