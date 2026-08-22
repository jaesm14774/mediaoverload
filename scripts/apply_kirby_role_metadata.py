"""Apply source-reviewed visual metadata to anime.anime_roles.

This is deliberately a Kirby data-seeding operation, not workflow logic. The
workflow remains character-agnostic; this script only replaces the Kirby rows'
description and keywords with source-grounded visual identity text.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
HARVEST_PATH = REPO_ROOT / "artifacts" / "kirby_role_research" / "20260822" / "source_harvest.json"
REPORT_PATH = REPO_ROOT / "artifacts" / "kirby_role_research" / "20260822" / "metadata_update_report.json"
TABLE_NAME = "anime.anime_roles"
ROLE_GROUP = "Kirby"


# Every entry below is an appearance-only rewrite grounded in the harvested
# Physical Appearance/intro text. Relationship, plot, game-title, and other
# character references are intentionally excluded from both fields.
CURATED: dict[str, tuple[str, str]] = {
    "Kirby": ("粉紅色、柔軟的圓球狀生物，兩隻短手、兩隻紅色腳，臉上有兩隻橢圓形眼睛與淡淡腮紅。", "粉紅色,圓球身體,短手,紅色腳,橢圓眼睛,腮紅"),
    "MetaKnight": ("圓球身體的蒙面劍士，穿銀色面具、深藍或深紫披風與金色裝飾，背後可展開蝙蝠翼，手持帶紅寶石的金色尖刺長劍。", "蒙面劍士,銀色面具,深藍披風,蝙蝠翼,金色長劍,紅寶石劍柄"),
    "KingDedede": ("藍色、身材圓胖的企鵝樣生物，黃色喙狀嘴、藍色眼睛，穿紅袍與紅色圓帽、黃色手套，腰間有紅黃鋸齒紋腰帶，手持巨大木槌。", "藍色,圓胖,企鵝樣,紅袍,紅帽,黃色手套,木槌,鋸齒腰帶"),
    "Waddle Dee": ("棕褐色梨形臉、栗色眼睛且沒有嘴巴的圓身生物，四肢短小，腳呈蜂蜜色，臉頰帶紅暈。", "棕褐色梨形臉,栗色眼睛,圓身,短手,蜂蜜色腳,紅暈"),
    "Waddle Doo": ("淺珊瑚色的圓身獨眼生物，眼睛從頭部中央凸出，眼上有兩束頭髮，手臂短小、腳呈淺橘色且沒有嘴巴。", "淺珊瑚色,圓身,獨眼,兩束頭髮,短手,淺橘色腳,無嘴"),
    "Wheelie": ("外形像灰色輪胎的生物，擁有兩隻大型亮眼與紅色眼皮；常見造型有紅色座板、把手與排氣管。", "灰色輪胎,大型眼睛,紅色眼皮,紅色座板,把手,排氣管"),
    "Bronto Burt": ("粉紅色球形飛行生物，長著細小的蒼蠅狀翅膀、橘色或黃色腳，臉上通常帶有皺眉表情。", "粉紅色,球形,飛行,細翅膀,橘色腳,皺眉表情"),
    "Sword Knight": ("身形細小的劍士，穿紫色裝甲、四道脊的長頭盔與大型肩甲，露出白色手套與藍色蛋形身體，使用銀色直刃金柄長劍。", "紫色裝甲,長頭盔,四道脊,大型肩甲,白手套,銀色長劍"),
    "Blade Knight": ("身形細小的劍士，穿綠色裝甲與長頭盔，頭頂有紅色流蘇、金色面罩與大型肩甲，臉部呈無表情粉紅色，使用銀色長劍。", "綠色裝甲,紅色流蘇,金色面罩,大型肩甲,粉紅臉,銀色長劍"),
    "Poppy Bros. Jr": ("帶有寬大笑容的精靈樣生物，身體與藍色軟帽同色，帽頂有白色絨球與白邊，衣服有兩顆黃色鈕扣，雙手可獨立漂浮，穿黃色尖鞋。", "精靈樣,藍色軟帽,白色絨球,黃色鈕扣,漂浮雙手,黃色尖鞋"),
    "Nago": ("圓胖的日本短尾貓樣生物，耳朵與尾巴為棕色，頭部與背部有橘色斑點，臉上有四根鬍鬚，四肢短小，眼睛通常閉合。", "圓胖貓,棕色耳朵,棕色尾巴,橘色斑點,四根鬍鬚,短四肢"),
    "Rick": ("圓身倉鼠，白色毛皮帶淺棕色頭部與背部斑塊，腳呈粉紅色，雙手白色，臉上有三角形粉紅鼻子與兩側鬍鬚。", "圓身倉鼠,白色毛皮,淺棕斑塊,粉紅腳,三角鼻子,鬍鬚"),
    "Gooey": ("小型藍色黏液狀生物，身體呈柔軟水滴或圓 blob 形，擁有兩隻圓形大眼睛與細長紅舌頭。", "藍色黏液,柔軟水滴形,圓形大眼,紅舌頭"),
    "Adeleine": ("年輕女孩，穿綠色有領畫家罩衫、灰色裙子與紅色貝雷帽，深色頭髮從帽下露出，手持藍色畫筆與棕色調色盤。", "年輕女孩,綠色罩衫,灰色裙子,紅色貝雷帽,藍色畫筆,棕色調色盤"),
    "Ribbon": ("身形嬌小的精靈女孩，留淡粉色蓬鬆及肩髮並繫大型紅色蝴蝶結，穿紅色長袖洋裝、白色荷葉領與棕色鞋，背後有半透明青色翅膀。", "嬌小精靈,淡粉色頭髮,紅色蝴蝶結,紅色洋裝,青色翅膀,棕色鞋"),
    "Dark Meta Knight": ("深灰色球形蒙面騎士，穿帶刮痕與尖角的面具、深色破損披風、深紅護脛與銀色長劍，眼睛呈淡黃色，劍柄嵌藍色寶石。", "深灰色,球形騎士,刮痕面具,破損披風,深紅護脛,銀劍,藍寶石"),
    "Dark Matter": ("黑色球體，中央有一隻眼睛；部分形態帶黑色斗篷、面具與長劍，另一種形態有多條紫色或橘色花瓣狀觸肢。", "黑色球體,中央獨眼,黑斗篷,面具,長劍,花瓣狀觸肢"),
    "Zero": ("巨大的白色球體，正面中央有一隻血紅色眼睛，眼睛具有深紅虹膜與細小黑色瞳孔。", "巨大白球,中央獨眼,血紅眼睛,深紅虹膜,黑色瞳孔"),
    "Zero Two": ("底部略扁的白色球體，中央有深紅色大眼與黑色虹膜、白色瞳孔，具分節翅膀、深紅羽毛、光環、頭部繃帶與可伸長的尖刺尾部。", "白色球體,深紅大眼,黑虹膜,分節翅膀,深紅羽毛,光環,繃帶,尖刺尾"),
    "Grand Doomer": ("巨大的液態黃色球形生物，周圍帶橘色光暈，底部有十三片羽毛狀突起，長著帶眼睛圖案的鳥翼、紅色眼睛與背部尖刺。", "巨大球形,液態黃色,橘色光暈,十三片羽毛,鳥翼,紅眼,背部尖刺"),
    "Marx": ("小型淡紫色小丑樣生物，穿紅藍分色且帶白色圖案的帽子與紅色蝴蝶結，腳穿棕色鞋，臉上有深紫大眼與固定笑容，持條紋球。", "淡紫色,小丑樣,紅藍分色帽,紅色蝴蝶結,棕色鞋,深紫大眼,條紋球"),
    "Blipper": ("魚樣生物，上半身偏紅、腹部橘色或黃色，身側與背部有淡藍色魚鰭，戴藍框黑帶護目鏡，嘴巴小而常張開。", "魚樣,紅色上身,橘色腹部,淡藍魚鰭,藍框護目鏡,黑帶,小嘴"),
    "Kabu": ("棕色頭部雕像，底部扁平、頂部圓弧，臉上有兩個深黑方形眼睛、中央小鼻子與下方張開的大嘴。", "棕色雕像,頭部形狀,扁平底部,黑色方眼,小鼻子,張嘴"),
    "Droppy": ("小型黃色圓身生物，四肢短小，外觀可呈現黃色、粉紅色或紅色變化。", "小型生物,黃色,圓身,短四肢,粉紅變化,紅色變化"),
    "Kracko": ("巨大的蓬鬆雲狀怪物，中央有一隻明亮獨眼，雲頂淺藍、雲底淺粉、中段白色，外圍環繞金色尖刺。", "巨大雲狀,中央獨眼,淺藍雲頂,淺粉雲底,白色雲身,金色尖刺"),
    "Gaw Gaw": ("棕色狼樣生物，耳朵尖、鼻子細小呈黑色，口鼻部與手背為淺棕色，雙手各有兩個鼴鼠樣爪子，眼睛總是閉著。", "棕色狼樣,尖耳,黑鼻子,淺棕口鼻,鼴鼠爪,閉眼"),
    "Parasol Waddle Dee": ("棕褐色圓身短肢生物，手持通常以白色與紅色或橘色交替條紋的陽傘，傘柄綠色、末端黃色。", "棕褐色,圓身,短肢,條紋陽傘,紅白傘面,綠色傘柄,黃色傘端"),
    "Flamer": ("紅色球形生物，只有一隻黑色且外圍帶白色的亮眼，身側有四個深紅色旋鈕，旋轉時外圍包覆火焰。", "紅色球形,單眼,黑眼白邊,四個旋鈕,火焰外罩"),
    "Kabula": ("大型黃色飛行砲台，外形像飛艇，常見造型有棕褐色拼布外殼、橘色鰭、前方砲管、後方渦輪與金屬鉚釘。", "黃色飛艇,飛行砲台,拼布外殼,橘色鰭,砲管,後方渦輪,鉚釘"),
    "Noddy": ("圓形粉紅色瞌睡生物，戴橘色睡帽，帽子有米色圓點、白色帽沿與白色絨球，擁有大黑眼、小嘴與兩隻橘色圓腳，沒有手臂。", "粉紅圓身,睡帽,米色圓點,白色絨球,黑眼,橘色圓腳,無手臂"),
    "Bonkers": ("大猩猩樣生物，沒有可見眼睛，穿深藍色破褲與開衩上衣，手臂肌肉發達，留紫色鬢角與蓬巴杜髮型，揮舞巨大木槌。", "猩猩樣,無可見眼睛,深藍破褲,紫色鬢角,蓬巴杜髮型,巨大木槌"),
    "Birdon": ("小型鸚鵡樣飛行生物，擁有大型藍眼睛、飛行員帽與護目鏡，頭頂有鮮豔羽毛冠，腳短小。", "鸚鵡樣,大型藍眼,飛行員帽,護目鏡,鮮豔羽毛冠,短腳"),
    "Cappy": ("外觀像會彈跳的蘑菇，蘑菇帽下藏著土偶樣臉部，卸下帽子後可見簡單的雕像狀頭部與眼睛。", "蘑菇外形,蘑菇帽,土偶樣臉,彈跳,雕像狀頭部"),
    "Galbo": ("小型無四肢紅色龍樣生物，腹部為白色或乳黃色，嘴巴呈鋸齒狀，背部排列橘色尖刺。", "紅色,無四肢,龍樣,乳黃色腹部,鋸齒嘴,橘色背刺"),
    "Pteran": ("翼龍樣生物，皮膚深紫色、腹部乳黃色，頭部有大型角狀冠，具長而似蝙蝠的翅膀，沒有可見腳。", "翼龍樣,深紫色,乳黃色腹部,角狀冠,蝙蝠翼,無可見腳"),
    "Sir Kibble": ("穿金色盔甲的小型騎士，黃色面罩下是黑色空洞，頭盔頂端有可投擲的刀刃，腳穿銀色護甲並戴紅色手套。", "金色盔甲,黃色面罩,黑色空洞,頭盔刀刃,銀色護腳,紅手套"),
    "Dark Mind": ("具有兩種外觀的黑暗生物：一種戴王冠、披深藍光環斗篷並露出橘色核心；另一種是帶紅色氣體光環的巨大橘色火球，中央有眼睛。", "王冠,深藍斗篷,橘色核心,巨大火球,中央獨眼,紅色光環"),
    "Axe Knight": ("戴骷髏面具與維京式頭盔的小型騎士，手持幾乎和身體一樣大的斧頭。", "骷髏面具,維京頭盔,小型騎士,巨大斧頭"),
    "Broom Hatter": ("黃色橢圓形生物，腳呈紅色、手臂短而鈍，戴黑色或藍色巫師帽與淡藍帽帶，手持掃帚，臉上沒有可見五官。", "黃色橢圓身,紅色腳,短手臂,巫師帽,淡藍帽帶,掃帚,無臉"),
    "Mumbies": ("漂浮的木乃伊頭部，身體包裹白色布條，布條中央留有孔洞露出一隻黃色或紅色眼睛。", "漂浮頭部,木乃伊,白色布條,眼部孔洞,黃色眼睛,紅色眼睛"),
    "Tac": ("小型圓身黃色生物，穿黑色竊賊服與涼鞋，頭上有內側藍色的貓耳，背著帶白色藤蔓圖案的綠色袋子。", "黃色圓身,黑色竊賊服,涼鞋,藍內耳,綠色袋子,白色藤蔓圖案"),
    "Kine": ("大型淺藍色太陽魚樣生物，身體呈半圓形，具有高大的黃色背鰭與臀鰭、蛤蜊形胸鰭，張開厚唇橘色大嘴。", "淺藍太陽魚,半圓身體,黃色背鰭,黃色臀鰭,蛤蜊形胸鰭,橘色大嘴"),
    "Coo": ("紫色羽毛的貓頭鷹，黃色喙、白色腹部與三道紫色羽毛條紋，翅膀黑色翼尖與白色內側，具有黃色爪子與頭頂羽毛尖刺。", "紫色貓頭鷹,黃色喙,白色腹部,三道條紋,黑色翼尖,黃色爪子,頭頂尖羽"),
    "Pitch": ("小型淺綠色幼鳥，翅膀末端為深綠色，腹部與翼下為白色，擁有黃色喙與兩個小圓眼。", "淺綠幼鳥,深綠翼尖,白色腹部,白色翼下,黃色喙,小圓眼"),
    "Nightmare": ("黑藍到深粉漸層的暗色球體，表面覆有銀色發光星點；另一形態穿星點旋渦長袍、銀色肩甲與單鏡片眼鏡，袍下露出扭曲黑暗核心。", "暗色球體,銀色星點,藍粉漸層,旋渦長袍,銀色肩甲,單鏡片眼鏡,黑暗核心"),
    "Magolor": ("矮小棕色異星生物，沒有腳，雙手與身體分離且戴奶油色手套，穿藍色金邊齒輪圖案長袍與白色披風圍巾，黃色橢圓眼睛，頭罩兩側有貓耳狀突起。", "棕色異星生物,無腳,漂浮雙手,奶油手套,藍色長袍,金邊齒輪,白披風,黃色眼睛"),
    "Star Dream": ("白色螺絲或圓柱形機械，頭部像螺絲，中央有圓形黃色玻璃眼，機體布滿粉紅色電路光線，兩側有金色裝飾的天使翼與藍色玻璃碎片。", "白色圓柱機械,螺絲頭,黃色玻璃眼,粉紅電路,天使翼,金色裝飾,藍色碎片"),
    "Hyness": ("穿白色長袍與兜帽的生物，衣物有金色滾邊與古代符號，白色面紗遮住大部分臉，露出藍色臉部、寬大鼻子、藍黃眼睛與橘色耳尖。", "白色長袍,兜帽,金色滾邊,古代符號,白面紗,藍色臉,寬鼻,橘色耳尖"),
    "Fecto Elfilis": ("青綠色皮毛的纖細生物，四肢極長、軀幹纖瘦，腹部與蓬鬆領圈為橘色，雙手有長紫色指甲，尾巴似長尾狐，頭上有向內彎曲的金色角與巨大的鋸齒狀翼耳。", "青綠皮毛,纖細身形,長四肢,橘色腹部,紫色長指甲,狐尾,金色彎角,翼耳"),
    "Chef Kawasaki": ("高挑的橢圓形橘色生物，手臂末端呈短 stub，眼睛細長、笑容彎曲，穿白色高帽與全身圍裙，胸前有藍色條紋口袋，手持煎鍋。", "橘色橢圓身,細長眼,白色廚師帽,全身圍裙,藍條紋口袋,煎鍋"),
    "Tiff": ("淺黃色皮膚的年輕女孩，留長金髮並束成華麗馬尾，綁紫色與橘色髮夾，擁有亮綠眼睛，穿粉紅上身與淺綠下身組成的鋸齒分界連身衣，腳穿橘鞋。", "淺黃色皮膚,金色馬尾,紫橘髮夾,亮綠眼睛,粉紅綠連身衣,橘鞋"),
    "Tuff": ("淺黃橘色皮膚的男孩，頭髮上半部為淺黃橘色、髮尖為叢林綠並覆蓋眼睛，只穿紫色短褲、斜向紅色吊帶與綠棕色鞋。", "淺黃橘皮膚,雙色頭髮,叢林綠髮尖,遮眼髮型,紫色短褲,紅色吊帶,綠棕鞋"),
    "Prince Fluff": ("蔚藍色的圓球身體，橘色腳、棕色眼睛，戴金色毛氈王冠，臉上保持皺眉表情。", "蔚藍色,圓球身體,橘色腳,棕色眼睛,金色毛氈王冠,皺眉"),
    "Mr. Frosty": ("沒有獠牙的類人海象，白色毛皮、淺藍色口鼻與臉頰，穿深藍或靛色吊帶褲，腳呈藍色或靛色，嘴裡有小尖牙。", "類人海象,白色毛皮,淺藍口鼻,深藍吊帶褲,靛色腳,小尖牙"),
    "Shadow Kirby": ("深灰色圓球身體，黑色腳與眼睛；部分造型呈半透明紫色，身體內部像漩渦般流動。", "深灰色,圓球身體,黑色腳,黑色眼睛,半透明紫色,漩渦身體"),
    "Galacta Knight": ("身體呈深粉紅色的裝甲騎士，穿白銀或銀色盔甲，戴有兩根彎曲金角的面具，背後有羽毛狀薰衣草色翅膀，手持粉紅長槍與白色小盾。", "深粉紅身體,銀色盔甲,雙金角面具,薰衣草翅膀,粉紅長槍,白色小盾"),
    "Bandana Waddle Dee": ("棕褐色梨形臉與圓身短肢生物，頭上戴海軍藍頭巾，手持長槍，皮膚常呈偏紅橘色。", "棕褐色,梨形臉,圓身,海軍藍頭巾,長槍,紅橘皮膚"),
    "Elfilin": ("覆蓋青綠色毛皮的栗鼠樣生物，巨大圓耳約為身體兩倍大，耳內呈橘黃相間，身體圓胖、四肢為短小突起，鼻子紅橘色，尾巴蓬鬆。", "青綠毛皮,栗鼠樣,巨大圓耳,橘黃耳內,圓胖身體,紅橘鼻,蓬鬆尾巴"),
    "Susie": ("身體小、頭部大且漂浮的女孩，白色臉部有大藍眼、睫毛與粉紅腮紅，留直順洋紅色長髮，穿淺灰色金屬套裝、深灰鉛筆裙與黃橘手套，手部與身體分離。", "漂浮頭部,白臉,大藍眼,洋紅長髮,淺灰金屬套裝,深灰裙,黃橘手套"),
    "Taranza": ("棕色蜘蛛樣生物，頭上有橘色尖角與銀色捲髮，臉與髮間分布多隻眼睛，穿深綠蛛網披風與深紅圍巾，底部有刺狀附肢並帶六隻漂浮戴手套的手。", "棕色蜘蛛樣,橘色尖角,銀色捲髮,多隻眼睛,蛛網披風,深紅圍巾,六隻漂浮手"),
    "Daroach": ("灰色老鼠樣生物，擁有大型耳朵、漂浮圓手與三根黃色長爪，戴紅色高禮帽與鋸齒紅斗篷，頸部有金色鈴鐺，眼睛呈黃色眼白與棕色虹膜。", "灰色老鼠,大型耳朵,漂浮圓手,黃色長爪,紅色高禮帽,紅斗篷,金色鈴鐺"),
    "Spinni": ("黃色老鼠樣生物，白色腹部、橘色腳、長口鼻與紅色鼻子，身形纖細，披長紅斗篷並戴大型亮紅色太陽眼鏡。", "黃色老鼠,白色腹部,橘色腳,長口鼻,紅鼻子,紅斗篷,紅色太陽眼鏡"),
    "Storo": ("藍色、身材壯碩的雙足老鼠樣生物，戴紅色頭巾與無袖上衣，臉有黑色圓鼻與突出單齒，眼睛被頭巾與眼罩遮住，腳短小且各有三趾。", "藍色壯碩,雙足老鼠,紅頭巾,無袖上衣,黑圓鼻,突出單齒,眼罩,三趾"),
    "Doc": ("小型藍白色老鼠樣生物，耳朵大而圓，臉部大部分被綠色鬍鬚覆蓋，常搭乘紅色飛碟，飛碟配有砲口與機械裝置。", "藍白老鼠,大圓耳,綠色大鬍鬚,紅色飛碟,砲口,機械裝置"),
    "Gorimondo": ("大型猿猴，灰色皮膚與黑色毛髮，手臂寬大、腿短小，鼻口部蓬鬆，頭頂白色莫霍克髮型，鼻環、肩上紅色條紋與臉頰紅色塗痕。", "大型猿猴,灰色皮膚,黑色毛髮,寬大手臂,白色莫霍克,鼻環,紅色肩紋,臉頰塗痕"),
    "Clawroline": ("金色毛皮的擬人豹，身上有棕色豹斑與米色胸毛，眼睛細長呈黃色並帶紫色虹膜，身材纖細、腰身明顯，尾巴末端棕色，爪尖塗紫色指甲。", "擬人豹,金色毛皮,棕色豹斑,米色胸毛,黃色眼睛,紫色眼影,棕尾尖,紫色爪尖"),
    "Sillydillo": ("巨大的犰狳，灰色甲殼上插有街道路牌，腹面白灰色，雙臂細長並以三根尖爪收尾，眼睛凸出且方向不一，紅鼻子、吐舌與兩顆門牙。", "巨大犰狳,灰色甲殼,街道路牌,白灰腹面,尖爪,凸眼,紅鼻子,吐舌,門牙"),
    "Leongar": ("肌肉發達的擬人獅，橘棕色身體、紅色鬃毛與棕色眼睛，眼周與上臂有橘色紋樣，戴黑色手環，常披紅色斗篷與白色褶邊。", "擬人獅,橘棕身體,紅鬃毛,棕眼,橘色紋樣,黑手環,紅斗篷,白褶邊"),
    "Awoofy": ("橘色毛皮的犬樣生物，腹部與口鼻為米色，耳朵尖、尾巴略蓬鬆，後腳黃色、鼻子亮黑色，耳尖尾端前腳與額頭帶紅色標記。", "橘色犬樣,米色腹部,尖耳,蓬鬆尾,黃色後腳,黑鼻子,紅色標記"),
    "Knuckle Joe": ("小型人形戰士，耳朵尖、頭髮金黃而尖刺，穿藍色連身服、白手套與藍鞋，戴白色頭帶，頭帶中央嵌紅色寶石，肩上有小型紅護甲。", "小型戰士,尖耳,金色刺髮,藍色連身服,白手套,藍鞋,白頭帶,紅寶石"),
    "Rocky": ("磚塊形石頭生物，身體由岩石構成，腳呈黃色或橘色，常見棕色外觀，也有藍色或綠色變體，部分造型戴藍白頭帶。", "磚塊形,岩石身體,黃色腳,橘色腳,棕色,藍色變體,綠色變體,頭帶"),
    "Gordo": ("小型藍色金屬球，擁有兩隻大眼睛與八根可同步伸縮的尖刺，尖刺通常呈金色或銀色。", "藍色金屬球,大眼睛,八根尖刺,可伸縮,金色尖刺,銀色尖刺"),
    "Scarfy": ("小型橘色貓樣生物，尖耳、黃色腮紅、亮藍色眼睛與寬笑容；變化形態呈米色、單眼、無腮紅並露出尖牙。", "橘色貓樣,尖耳,黃色腮紅,亮藍眼,寬笑容,單眼,尖牙,變化形態"),
    "Sparky": ("綠色淚滴狀黏液生物，身體底部有小黑眼，兩側有被電氣包圍的淡黃綠色球體。", "綠色淚滴形,黏液身體,底部黑眼,淡黃綠球體,電氣光芒"),
    "Bio Spark": ("身形小而圓的忍者，身體包在背後繫結的斗篷中，臉被遮住只露白色橢圓眼睛，頭頂有金色環形頭飾與長紅流蘇，手腳呈紅色。", "小型忍者,圓身,斗篷,白色橢圓眼,金色頭飾,紅色流蘇,紅手紅腳"),
}


def _connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=os.getenv("mysql_host"),
        port=int(os.getenv("mysql_port", "3306")),
        user=os.getenv("mysql_user"),
        password=os.getenv("mysql_password"),
        database=os.getenv("mysql_db_name"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        read_timeout=30,
        write_timeout=30,
        autocommit=False,
    )


def main() -> None:
    load_dotenv(REPO_ROOT / "media_overload.env")
    harvest = json.loads(HARVEST_PATH.read_text(encoding="utf-8"))
    source_rows = {
        str(row["role_name_en"]): row
        for row in harvest["records"]
        if row.get("resolution") == "exact_or_alias"
    }

    connection = _connect()
    report: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "group_name": ROLE_GROUP,
        "curated_count": len(CURATED),
        "cleared_unverified_count": 0,
        "updated": [],
        "cleared_unverified": [],
    }
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT id, role_name_en, role_description, keywords, status "
                f"FROM {TABLE_NAME} WHERE group_name=%s ORDER BY id FOR UPDATE",
                (ROLE_GROUP,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
            db_names = {str(row.get("role_name_en") or "") for row in rows}
            missing = sorted(set(CURATED) - db_names)
            if missing:
                raise RuntimeError(f"Curated role names are missing from DB: {missing}")

            for row in rows:
                name = str(row.get("role_name_en") or "")
                if name in CURATED:
                    if name not in source_rows:
                        raise RuntimeError(f"Curated role lacks exact source evidence: {name}")
                    description, keywords = CURATED[name]
                    if len(description) > 1024 or len(keywords) > 512:
                        raise RuntimeError(f"Field length exceeded for {name}")
                    cursor.execute(
                        f"UPDATE {TABLE_NAME} SET role_description=%s, keywords=%s WHERE id=%s AND group_name=%s",
                        (description, keywords, row["id"], ROLE_GROUP),
                    )
                    report["updated"].append(
                        {
                            "id": row["id"],
                            "role_name_en": name,
                            "source_title": source_rows[name].get("source_title"),
                            "source_url": source_rows[name].get("source_url"),
                            "status_after": row["status"],
                        }
                    )
                else:
                    # Existing text is not source-verified. Empty metadata plus
                    # status=-1 prevents it from entering a generation prompt.
                    cursor.execute(
                        f"UPDATE {TABLE_NAME} SET role_description=NULL, keywords=NULL, status=-1 "
                        f"WHERE id=%s AND group_name=%s",
                        (row["id"], ROLE_GROUP),
                    )
                    report["cleared_unverified"].append(
                        {
                            "id": row["id"],
                            "role_name_en": name,
                            "status_before": row["status"],
                        }
                    )
            connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    report["cleared_unverified_count"] = len(report["cleared_unverified"])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "updated": len(report["updated"]),
        "cleared_unverified": report["cleared_unverified_count"],
        "report": str(REPORT_PATH),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
