# -*- coding: utf-8 -*-
"""seed_products

内存商品库的种子数据：跨境 SPU，覆盖旅行装备、数码配件、家居、户外等品类，
中文标题 + 关键词化描述，便于关键词召回命中"抗造 / 轻便 / 不要塑料"这类口语属性词。

分两部分：

    前 10 个 SPU（P1001~P1010）逐字段展开写，保持不变——多个单测依赖它们的 id 与排序；
    P1011 起为**召回评测语料扩充**（见 13-2 章），用紧凑表驱动写法控制篇幅。

为何要扩充：10 个 SPU 下 Recall@10 恒等于 1，指标没有区分度，评测形同虚设。
扩充的唯一目的是**造出区分度**：同品类多候选、标题相近但材质/价位不同、
含易混淆项（"旅行三件套"vs"毛巾三件装"）、价格与 ships_to 分布拉开。

注意：新增语料刻意避开与 P1001 在"旅行三件套 抗造"这条 query 上正面相撞
（竞品 P1011 不写"抗造"），以免推翻已有单测的 top-1 断言。
"""
from __future__ import annotations

from app.domain.catalog.money import Money
from app.domain.catalog.product import Product, ProductHighlight
from app.domain.catalog.sku import Sku


def _sku(sku_id: str, spec: str, major: float, currency: str, stock: int) -> Sku:
    return Sku(sku_id=sku_id, spec=spec, price=Money.from_major_units(major, currency), stock=stock)


# 紧凑表：(id, 标题, 品牌, 品类, 产地, 描述, ships_to, [(sku后缀, 规格, 价, 币种, 库存)], [(亮点名, 亮点值)])
_EXTRA_SPECS: list[tuple] = [
    # ---- 旅行装备：三件套 / 单品 / 箱包 多候选，制造排序区分度 ----
    ("P1011", "Voyager 旅行三件套 记忆棉款", "Voyager", "旅行装备", "CN",
     "记忆棉颈枕 遮光眼罩 收纳袋 三件套 轻便 长途飞行 出差 旅行必备",
     ["CN", "US", "SG"], [("S1", "深灰", 139.0, "CNY", 60)], [("材质", "记忆棉 + 涤纶外套")]),
    ("P1012", "NestRest 充气颈枕 按压式", "NestRest", "旅行装备", "CN",
     "充气颈枕 按压充气 收纳体积小 飞机 高铁 侧睡支撑 轻便 180g",
     ["CN", "US"], [("S1", "海盐蓝", 69.0, "CNY", 120)], [("体积", "收纳后仅掌心大小")]),
    ("P1013", "BlackoutPro 3D 真丝遮光眼罩", "BlackoutPro", "旅行装备", "CN",
     "真丝 遮光眼罩 3D立体 不压眼 天然材质 无塑料 睡眠 飞机 高铁",
     ["CN", "US", "EU"], [("S1", "豆沙色", 89.0, "CNY", 90)], [("材质", "6A 桑蚕丝，无塑料")]),
    ("P1014", "PackMate 压缩收纳袋 6件套", "PackMate", "旅行装备", "CN",
     "压缩收纳袋 六件套 分装 行李整理 防潮 可重复使用 轻便",
     ["CN", "US", "SG"], [("S1", "灰蓝混色", 99.0, "CNY", 140)], [("套装", "大中小共 6 只")]),
    ("P1015", "TrailOx 24寸托运行李箱 铝框款", "TrailOx", "旅行装备", "DE",
     "铝镁合金框 托运尺寸 24寸 结实抗摔 万向静音轮 TSA锁 长途旅行",
     ["CN", "US", "EU"], [("S1", "银色 / 24寸", 1199.0, "CNY", 12)], [("箱体", "铝框，抗摔")]),
    ("P1016", "GlideCase 20寸登机箱 PC硬壳", "GlideCase", "旅行装备", "CN",
     "PC聚碳酸酯硬壳 塑料箱体 登机尺寸 20寸 轻量 万向轮 平价",
     ["CN"], [("S1", "碳黑 / 20寸", 399.0, "CNY", 45)], [("箱体", "PC 塑料硬壳")]),
    ("P1017", "Wanderlite 轻量旅行腰包", "Wanderlite", "旅行装备", "KR",
     "腰包 胸包 轻量 120g 防泼水 贴身 杂物袋 日常通勤 旅行",
     ["CN", "JP", "SG"], [("S1", "雾霞蓝", 79.0, "CNY", 100)], [("重量", "仅 120g")]),
    ("P1018", "DryPack 30L 防水卷口背包", "DryPack", "旅行装备", "CN",
     "卷口防水 30L TPU涂层 户外 涉水 结实抗造 骑行 满水不漏",
     ["CN", "US", "EU"], [("S1", "墨绿", 219.0, "CNY", 55)], [("防水", "IPX6 卷口密封")]),
    ("P1019", "LinenFold 亚麻旅行衣物收纳套", "LinenFold", "旅行装备", "CN",
     "亚麻 天然材质 无塑料 透气 衣物分装 折叠 环保 小众设计",
     ["CN", "JP"], [("S1", "本色三只装", 129.0, "CNY", 70)], [("材质", "100% 亚麻，无塑料")]),
    ("P1020", "SoleFresh 折叠旅行拖鞋", "SoleFresh", "旅行装备", "CN",
     "折叠拖鞋 便携袋 酒店 飞机 轻便 160g 可水洗",
     ["CN", "US"], [("S1", "灰色 M", 49.0, "CNY", 180)], [("重量", "双足仅 160g")]),
    ("P1021", "AquaLite 折叠硅胶水壶 600ml", "AquaLite", "旅行装备", "CN",
     "食品级硅胶 折叠水壶 600ml 轻便 徒步 飞机 可收纳",
     ["CN", "US", "EU"], [("S1", "霉绿", 59.0, "CNY", 160)], [("收纳", "折叠后厚 3cm")]),
    ("P1051", "ZipPouch RFID 证件收纳包", "ZipPouch", "旅行装备", "CN",
     "RFID 防盗刷 证件收纳 护照夹 登机牌 多卡位 轻薄",
     ["CN", "US", "EU"], [("S1", "碳黑", 69.0, "CNY", 130)], [("安全", "RFID 屏蔽层")]),
    ("P1052", "TravelScale 便携行李称", "TravelScale", "旅行装备", "CN",
     "行李称 50kg 电子称 便携 避免超重 背带式",
     ["CN", "US"], [("S1", "黑色", 45.0, "CNY", 200)], [("量程", "最大 50kg")]),
    ("P1053", "QuietEar 硅胶隔音耳塞", "QuietEar", "旅行装备", "CN",
     "硅胶耳塞 隔音降噪 32dB 睡眠 飞机 可水洗 带收纳盒",
     ["CN", "US", "EU"], [("S1", "透明三对装", 39.0, "CNY", 220)], [("降噪", "SNR 32dB")]),
    ("P1058", "SteamRest 蒸汽热敷眼罩", "SteamRest", "旅行装备", "JP",
     "蒸汽眼罩 发热 一次性 缓解眼部疲劳 睡眠 日本制",
     ["CN", "JP"], [("S1", "无香 14片", 79.0, "CNY", 110)], [("发热", "40℃ 持续 20 分钟")]),
    ("P1059", "CompressCube 压缩收纳方块", "CompressCube", "旅行装备", "CN",
     "压缩方块 双拉链 行李分装 省空间 高弹尼龙 轻便",
     ["CN", "US"], [("S1", "黑色两只装", 89.0, "CNY", 95)], [("压缩", "体积减少 40%")]),
    ("P1060", "UniversalStrap 行李捆扎带 TSA锁", "UniversalStrap", "旅行装备", "CN",
     "捆扎带 TSA 密码锁 防护 易辨识 加固行李箱",
     ["CN", "US", "EU"], [("S1", "橙色", 55.0, "CNY", 150)], [("锁具", "TSA 海关认可")]),
    ("P1049", "BudgetPack 20L 简易背包", "BudgetPack", "旅行装备", "CN",
     "20L 入门背包 轻便 平价 涤纶 日常 通勤",
     ["CN"], [("S1", "黑色", 39.0, "CNY", 300)], [("价位", "入门平价款")]),
    ("P1050", "LuxeTrunk 铝镁合金旅行箱 全铝款", "LuxeTrunk", "旅行装备", "DE",
     "全铝镁合金箱体 高级 结实抗摔 终身保修 商务 高价位",
     ["CN", "EU"], [("S1", "原色 / 26寸", 2999.0, "CNY", 5)], [("箱体", "全铝镁合金")]),

    # ---- 数码配件：同品牌高/低配 + 降噪竞品，测同类排序 ----
    ("P1022", "AeroHush Lite 半入耳蓝牙耳机", "AeroHush", "数码配件", "CN",
     "半入耳 蓝牙 5.3 轻便 长续航 30小时 通话降噪 入门价位",
     ["CN", "US"], [("S1", "白色", 299.0, "CNY", 80)], [("续航", "含仓 30 小时")]),
    ("P1023", "SilentBuds 主动降噪耳塞式耳机", "SilentBuds", "数码配件", "US",
     "主动降噪 ANC 深降噪 42dB 耳塞式 通勤 飞机 降噪耳机",
     ["CN", "US", "EU"], [("S1", "子夜蓝", 189.0, "USD", 30)], [("降噪", "深度 42dB")]),
    ("P1024", "VoltTrek 30W 迷你充电器", "VoltTrek", "数码配件", "CN",
     "30W 氮化镓 迷你 单口 折叠插脚 轻便 手机快充",
     ["CN", "US", "EU", "JP"], [("S1", "白色", 89.0, "CNY", 150)], [("体积", "仅鸡蛋大小")]),
    ("P1025", "VoltTrek 100W 四口氮化镓充电器", "VoltTrek", "数码配件", "CN",
     "100W 氮化镓 四口 同时充笔记本 全球插脚 出差 旅行",
     ["CN", "US", "EU", "JP"], [("S1", "深空灰", 299.0, "CNY", 60)], [("功率", "100W 四口分配")]),
    ("P1026", "PowerCore 10000mAh 移动电源", "PowerCore", "数码配件", "CN",
     "10000mAh 移动电源 锂电池 可登机 轻薄 自带线 锂电池空运限制",
     ["CN"], [("S1", "黑色", 129.0, "CNY", 90)], [("空运", "含锂电池，仅国内配送")]),
    ("P1027", "CableRoll 三合一磁吸数据线", "CableRoll", "数码配件", "CN",
     "三合一 数据线 磁吸收纳 Type-C Lightning 微口 快充 旅行",
     ["CN", "US", "EU"], [("S1", "银灰", 59.0, "CNY", 200)], [("兼容", "三种接口一线搞定")]),
    ("P1028", "GlobeAdapt 全球通用转换插头", "GlobeAdapt", "数码配件", "CN",
     "全球通用 转换插头 150国 内置USB 出国 旅行 安全门",
     ["CN", "US", "EU", "JP"], [("S1", "白色", 99.0, "CNY", 130)], [("兼容", "覆盖 150+ 国家")]),
    ("P1029", "ClearShot 铝合金手机三脚架", "ClearShot", "数码配件", "CN",
     "铝合金 三脚架 便携 蓝牙快门 旅拍 Vlog 结实",
     ["CN", "US"], [("S1", "黑色", 149.0, "CNY", 70)], [("材质", "铝合金管体")]),
    ("P1030", "SDVault 高速存储卡读卡器", "SDVault", "数码配件", "CN",
     "读卡器 SD TF 高速 USB3.2 带卡仓 摄影 便携",
     ["CN", "US", "EU"], [("S1", "银色", 79.0, "CNY", 110)], [("速率", "USB 3.2 Gen1")]),
    ("P1054", "BreezeNeck 便携挂颈风扇", "BreezeNeck", "数码配件", "CN",
     "挂颈风扇 无叶 三档风速 续航8小时 夏季 户外 便携",
     ["CN", "SG"], [("S1", "白色", 119.0, "CNY", 85)], [("续航", "最高 8 小时")]),

    # ---- 家居生活：“中性气质咖啡杯”这类气质类 query 的多候选 ----
    ("P1031", "TerraCotta 手工粗陶马克杯", "TerraCotta", "家居生活", "JP",
     "手工粗陶 马克杯 窑烧 天然材质 无塑料 侘寂 中性 小众 咖啡杯",
     ["CN", "JP"], [("S1", "灶变色", 128.0, "CNY", 40)], [("材质", "粗陶手作，无塑料")]),
    ("P1032", "Nordic 中性色陶瓷咖啡杯", "Nordic", "家居生活", "CN",
     "陶瓷 咖啡杯 中性色 北欧极简 磨砂釉 天然 无塑料 办公 居家",
     ["CN", "US", "EU"], [("S1", "雾霞灰", 89.0, "CNY", 120)], [("风格", "北欧中性色")]),
    ("P1033", "BambooLine 竹制餐具旅行套装", "BambooLine", "家居生活", "CN",
     "竹制餐具 筷勺叉 旅行套装 天然材质 无塑料 环保 便携袋",
     ["CN", "US", "EU"], [("S1", "原色", 69.0, "CNY", 160)], [("材质", "天然楱竹，无塑料")]),
    ("P1034", "LinenHome 亚麻抱枕套", "LinenHome", "家居生活", "CN",
     "亚麻 抱枕套 天然材质 透气 中性色 极简 居家",
     ["CN", "JP"], [("S1", "米白 45x45", 79.0, "CNY", 90)], [("材质", "水洗亚麻")]),
    ("P1035", "AromaStone 陶石香薰扩香器", "AromaStone", "家居生活", "CN",
     "陶石 扩香 无火 香薰 天然材质 小众设计 卧室",
     ["CN"], [("S1", "白色", 99.0, "CNY", 60)], [("方式", "无火被动扩香")]),
    ("P1036", "GlassPour 双层玻璃手冲壶", "GlassPour", "家居生活", "CN",
     "双层玻璃 手冲壶 隔热 咖啡 600ml 易清洗",
     ["CN", "US"], [("S1", "透明", 159.0, "CNY", 45)], [("容量", "600ml 双层隔热")]),
    ("P1037", "CorkMat 软木隔热垫", "CorkMat", "家居生活", "PT",
     "软木 隔热垫 天然材质 无塑料 餐桌 四只装",
     ["CN", "EU"], [("S1", "原色四只", 59.0, "CNY", 140)], [("材质", "葡萄牙软木")]),
    ("P1055", "SteamFree 便携挂烫机", "SteamFree", "家居生活", "CN",
     "手持挂烫机 便携 出差 蒸汽熨斗 快热 30秒 旅行",
     ["CN"], [("S1", "白色", 199.0, "CNY", 50)], [("预热", "30 秒出蒸汽")]),
    ("P1056", "ShoeBag 防水鞋袋两只装", "ShoeBag", "家居生活", "CN",
     "鞋袋 防水 加厚 行李分装 可重复使用 旅行",
     ["CN", "US"], [("S1", "灰色两只", 35.0, "CNY", 240)], [("容量", "单只可装 45 码")]),
    ("P1057", "MiniUmbrella 五折超轻晴雨伞", "MiniUmbrella", "家居生活", "CN",
     "五折伞 超轻 180g 防晒 UPF50+ 晴雨兼用 口袋伞",
     ["CN", "US", "JP"], [("S1", "黑胶", 89.0, "CNY", 130)], [("重量", "仅 180g")]),

    # ---- 户外运动：登山杖/露营灯/睡袋 各成梯度 ----
    ("P1038", "LumenGo Mini 钥匙扣手电", "LumenGo", "户外运动", "CN",
     "钥匙扣手电 便携 Type-C充电 强光 应急 轻 28g",
     ["CN", "US", "EU"], [("S1", "银色", 45.0, "CNY", 200)], [("重量", "仅 28g")]),
    ("P1039", "SolarLamp 太阳能营地灯", "SolarLamp", "户外运动", "CN",
     "太阳能 营地灯 露营灯 可充电 防水IPX5 可折叠 应急照明",
     ["CN", "US", "EU"], [("S1", "黑色", 129.0, "CNY", 75)], [("供电", "太阳能 + USB 双模")]),
    ("P1040", "CascadePro 铝合金折叠登山杖一对", "CascadePro", "户外运动", "CN",
     "铝合金 折叠 登山杖 快锁 减震 徒步 入门价位 320g单支",
     ["CN", "US"], [("S1", "黑色一对", 199.0, "CNY", 60)], [("材质", "7075 铝合金")]),
    ("P1047", "TrekPole 碳纤维登山杖一对", "TrekPole", "户外运动", "CN",
     "碳纤维 登山杖 超轻 190g单支 三节 徒步 登山 专业",
     ["CN", "US", "EU"], [("S1", "碳纤纹一对", 459.0, "CNY", 30)], [("重量", "单支仅 190g")]),
    ("P1041", "TrailSeat 超轻折叠露营凳", "TrailSeat", "户外运动", "CN",
     "折叠凳 超轻 680g 铝合金 露营 徒步 承重120kg 结实抗造",
     ["CN", "US"], [("S1", "墨绿", 169.0, "CNY", 65)], [("承重", "最大 120kg")]),
    ("P1042", "ThermoRest 自动充气防潮垫", "ThermoRest", "户外运动", "CN",
     "自动充气 防潮垫 露营 帐篷 保温 R值3.2 可拼接",
     ["CN", "US", "EU"], [("S1", "橙色单人", 279.0, "CNY", 40)], [("保温", "R 值 3.2")]),
    ("P1043", "SummitBag -5℃ 羽绒睡袋", "SummitBag", "户外运动", "CN",
     "羽绒睡袋 -5℃ 鸭绒 露营 保温 可压缩 1.2kg 信封式",
     ["CN", "US", "EU"], [("S1", "深蓝", 699.0, "CNY", 25)], [("适用", "舒适温 -5℃")]),
    ("P1044", "GripLine 静力登山绳 10mm", "GripLine", "户外运动", "CN",
     "静力绳 10mm 登山绳 承重 结实抗造 攀岩 户外安全",
     ["CN", "EU"], [("S1", "30米", 389.0, "CNY", 20)], [("强度", "断裂拉力 22kN")]),
    ("P1045", "HydroFlow 316不锈钢保温水壶", "HydroFlow", "户外运动", "CN",
     "316不锈钢 保温水壶 750ml 保温12小时 运动 户外 无塑料内胆",
     ["CN", "US", "EU"], [("S1", "碳黑 750ml", 189.0, "CNY", 80)], [("内胆", "316 不锈钢，无塑料")]),
    ("P1046", "WindShell 超轻防风外套", "WindShell", "户外运动", "CN",
     "防风外套 超轻 180g 可收纳 拨水 徒步 露营 男女同款",
     ["CN", "US", "EU"], [("S1", "炭黑 L", 259.0, "CNY", 55)], [("重量", "仅 180g 可压缩")]),
    ("P1048", "CampCook 钛合金炊具套装", "CampCook", "户外运动", "CN",
     "钛合金 炊具 露营锅 超轻 240g 可嵌套 无涂层 徒步",
     ["CN", "US", "EU"], [("S1", "钛原色两件", 429.0, "CNY", 28)], [("材质", "纯钛，无涂层")]),
]


def _build_extra() -> list[Product]:
    """把紧凑表展开成 Product。"""
    products: list[Product] = []
    for pid, title, brand, category, origin, desc, ships_to, skus, highlights in _EXTRA_SPECS:
        products.append(
            Product(
                product_id=pid,
                title=title,
                brand=brand,
                category=category,
                origin_country=origin,
                description=desc,
                highlights=[ProductHighlight(name, value) for name, value in highlights],
                ships_to=list(ships_to),
                skus=[
                    _sku(f"{pid}-{suffix}", spec, major, currency, stock)
                    for suffix, spec, major, currency, stock in skus
                ],
            ),
        )
    return products


def build_seed_products() -> list[Product]:
    return [
        Product(
            product_id="P1001",
            title="Nomadica 旅行三件套（收纳袋+颈枕+眼罩）",
            brand="Nomadica",
            category="旅行装备",
            origin_country="VN",
            description="帆布加尼龙材质 结实耐磨 抗造 轻便 无塑料感 小众设计师品牌 适合长途飞行 旅行收纳",
            highlights=[
                ProductHighlight("材质", "帆布+再生尼龙，非塑料"),
                ProductHighlight("重量", "全套 420g 轻便"),
                ProductHighlight("风格", "小众设计师联名款"),
            ],
            ships_to=["CN", "US", "SG"],
            skus=[
                _sku("P1001-S1", "军绿色", 189.0, "CNY", 50),
                _sku("P1001-S2", "沙漠黄", 199.0, "CNY", 30),
            ],
        ),
        Product(
            product_id="P1002",
            title="TrailOx 20寸登机行李箱 铝框款",
            brand="TrailOx",
            category="旅行装备",
            origin_country="DE",
            description="铝镁合金框架 PC箱体 结实抗摔 抗造 万向静音轮 TSA海关锁 登机尺寸 商务旅行",
            highlights=[
                ProductHighlight("箱体", "德国工艺铝框，抗摔"),
                ProductHighlight("轮组", "日本静音万向轮"),
            ],
            ships_to=["CN", "US", "EU"],
            skus=[
                _sku("P1002-S1", "银色 / 20寸", 899.0, "CNY", 20),
                _sku("P1002-S2", "黑色 / 20寸", 899.0, "CNY", 15),
            ],
        ),
        Product(
            product_id="P1003",
            title="Wanderlite 折叠旅行双肩包 35L",
            brand="Wanderlite",
            category="旅行装备",
            origin_country="KR",
            description="防泼水尼龙 超轻 可折叠收纳 大容量 35升 徒步 城市通勤 便宜实惠 高性价比",
            highlights=[
                ProductHighlight("重量", "仅 380g 超轻"),
                ProductHighlight("收纳", "可折叠成手掌大小"),
            ],
            ships_to=["CN", "JP", "SG"],
            skus=[
                _sku("P1003-S1", "石墨黑", 129.0, "CNY", 80),
                _sku("P1003-S2", "雾霾蓝", 129.0, "CNY", 60),
            ],
        ),
        Product(
            product_id="P1004",
            title="AeroHush 主动降噪蓝牙耳机 Pro",
            brand="AeroHush",
            category="数码配件",
            origin_country="US",
            description="主动降噪 蓝牙5.4 40小时续航 通话降噪 飞行旅行伴侣 头戴式 折叠便携",
            highlights=[
                ProductHighlight("降噪", "-45dB 深度主动降噪"),
                ProductHighlight("续航", "40 小时长续航"),
            ],
            ships_to=["CN", "US", "EU"],
            skus=[
                _sku("P1004-S1", "曜石黑", 219.0, "USD", 40),
                _sku("P1004-S2", "月光白", 229.0, "USD", 25),
            ],
        ),
        Product(
            product_id="P1005",
            title="VoltTrek 65W 氮化镓旅行充电器（全球插脚）",
            brand="VoltTrek",
            category="数码配件",
            origin_country="CN",
            description="氮化镓 GaN 65W 快充 全球通用插脚 英标欧标美标 出国旅行 多口 Type-C 轻巧",
            highlights=[
                ProductHighlight("插脚", "全球 150+ 国家通用"),
                ProductHighlight("功率", "65W 双口快充"),
            ],
            ships_to=["CN", "US", "EU", "JP"],
            skus=[
                _sku("P1005-S1", "标准版", 159.0, "CNY", 100),
            ],
        ),
        Product(
            product_id="P1006",
            title="TerraCotta 手工粗陶旅行茶具套装",
            brand="TerraCotta",
            category="家居生活",
            origin_country="JP",
            description="手工粗陶 一壶两杯 便携旅行装 无塑料 天然材质 小众手作 茶道 送礼",
            highlights=[
                ProductHighlight("材质", "天然粗陶，无塑料"),
                ProductHighlight("工艺", "日本手作窑烧"),
            ],
            ships_to=["CN", "JP"],
            skus=[
                _sku("P1006-S1", "原色", 268.0, "CNY", 18),
            ],
        ),
        Product(
            product_id="P1007",
            title="PeakDry 速干旅行毛巾三件装",
            brand="PeakDry",
            category="旅行装备",
            origin_country="TW",
            description="超细纤维 速干 轻薄 抗菌 三条装 大中小 游泳 健身 户外 便宜 高性价比",
            highlights=[
                ProductHighlight("速干", "3 分钟拧干即用"),
                ProductHighlight("装量", "大中小三条装"),
            ],
            ships_to=["CN", "US", "SG"],
            skus=[
                _sku("P1007-S1", "灰蓝绿三色", 79.0, "CNY", 200),
            ],
        ),
        Product(
            product_id="P1008",
            title="LumenGo 便携露营灯 可充电",
            brand="LumenGo",
            category="户外运动",
            origin_country="CN",
            description="露营灯 三档调光 Type-C充电 磁吸挂钩 防水 IPX5 户外 应急 停电 抗造耐摔",
            highlights=[
                ProductHighlight("防护", "IPX5 防水，抗摔"),
                ProductHighlight("续航", "最长 72 小时"),
            ],
            ships_to=["CN", "US", "EU"],
            skus=[
                _sku("P1008-S1", "军绿", 89.0, "CNY", 150),
                _sku("P1008-S2", "橙色", 89.0, "CNY", 90),
            ],
        ),
        Product(
            product_id="P1009",
            title="SilkRoute 桑蚕丝旅行睡袋内胆",
            brand="SilkRoute",
            category="旅行装备",
            origin_country="CN",
            description="100%桑蚕丝 亲肤 隔脏 超轻 200g 卷收便携 酒店青旅 露营 天然材质 无塑料",
            highlights=[
                ProductHighlight("材质", "100% 桑蚕丝，天然无塑料"),
                ProductHighlight("重量", "仅 200g"),
            ],
            ships_to=["CN", "US", "EU", "JP"],
            skus=[
                _sku("P1009-S1", "本白", 329.0, "CNY", 35),
            ],
        ),
        Product(
            product_id="P1010",
            title="CascadePro 钛合金折叠登山杖一对",
            brand="CascadePro",
            category="户外运动",
            origin_country="US",
            description="钛合金 折叠五节 快锁 减震 徒步 登山 结实抗造 轻量 260g单支 专业户外",
            highlights=[
                ProductHighlight("材质", "航空钛合金，结实抗造"),
                ProductHighlight("折叠", "五节折叠仅 36cm"),
            ],
            ships_to=["US", "CN", "EU"],
            skus=[
                _sku("P1010-S1", "钛原色一对", 149.0, "USD", 22),
            ],
        ),
        *_build_extra(),
    ]
