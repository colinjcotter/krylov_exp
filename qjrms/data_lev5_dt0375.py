# peddle plots data for lev5_dt0375

def input_lev5_dt0375(etadict, udict, ppp, dts=45):
    if ppp == 4:
        # norms dumped every 0.25 day

        if dts == 22.5:

            # AW0.375
            etadict[0.375] = [0.0002764668621849686, 0.0003023548847734355, 0.00031330744288794775, 0.0003211613260047293]
            udict[0.375] = [0.00036065475086278023, 0.000385203534511741, 0.0003973900542040997, 0.00040668959691644396]

            # AW0.4125
            etadict[0.4125] = [0.00024295125457962267, 0.0002675897153312497, 0.00027715478474604847, 0.00028268620963743744]
            udict[0.4125] = [0.00032921669103865155, 0.00035447796452342317, 0.0003679072966651552, 0.0003782028975484377]

            # AW0.45
            etadict[0.45] = [0.00021705602264140704, 0.00023994585010858903, 0.00024903572496854347, 0.0002542729065495491]
            udict[0.45] = [0.0003095606471221197, 0.00033594425810508035, 0.0003524296364224106, 0.00036667245188924105]

            # AW0.4875
            etadict[0.4875] = [0.00019643519778602433, 0.00021700107947011712, 0.00022574126927882686, 0.00023101401551628816]
            udict[0.4875] = [0.0002995452050282037, 0.00032747942301477917, 0.0003479408328839486, 0.000367178211170296]

            # AW0.525
            etadict[0.525] = [0.0001796267218826777, 0.00019759747971534464, 0.00020602815949989024, 0.00021137406727763622]
            udict[0.525] = [0.00029792603538442294, 0.0003280893695857168, 0.00035320146224005817, 0.0003780094055014392]

            # AW0.5625
            etadict[0.5625] = [0.0001661504662202764, 0.00018157661870095887, 0.00018982739012478653, 0.00019529391646582546]
            udict[0.5625] = [0.00030399123291961067, 0.0003372207787369675, 0.00036751766199434274, 0.00039824489234316265]

            # AW0.6
            etadict[0.6] = [0.00015628391720596985, 0.00016943288648364188, 0.00017772484488993488, 0.00018343204752984025]
            udict[0.6] = [0.0003171244618903795, 0.0003542538367078116, 0.0003901357154264304, 0.0004269938118179022]

            # AW0.6375
            etadict[0.6375] = [0.00015057947215279486, 0.00016183750295740466, 0.00017040911846206094, 0.00017648225193121856]
            udict[0.6375] = [0.0003365328957962195, 0.0003782819345210976, 0.0004200340123677045, 0.0004631578687509413]

            # AW0.675
            etadict[0.675] = [0.00014934650784075693, 0.0001591659331139501, 0.00016821134839921422, 0.00017472862723607203]
            udict[0.675] = [0.0003612410942718111, 0.0004081979451026498, 0.0004560444950064276, 0.0005055502546715767]

            # AW0.7125
            etadict[0.7125] = [0.00015236623800948527, 0.0001612270592600451, 0.0001708839372483343, 0.0001778738099407952]
            udict[0.7125] = [0.0003902233365783387, 0.0004428770700061947, 0.0004970456766754519, 0.0005530795915784146]

            # AW0.75
            etadict[0.75] = [0.00015896611591199668, 0.0001673291696037965, 0.00017769920999389333, 0.00018516900078154025]
            udict[0.75] = [0.00042255385647296036, 0.00048133011832148935, 0.0005420799456578253, 0.0006048437821811413]

            # AW0.825
            etadict[0.825] = [0.00017963128227777716, 0.00018811252088772414, 0.00020017124250969843, 0.00020862046660097297]
            udict[0.825] = [0.0004944861928530654, 0.0005666527746912619, 0.0006414962983905907, 0.0007185336764244586]

            # AW0.9
            etadict[0.9] = [0.0002061412183397661, 0.00021568278879262613, 0.00022975175846708034, 0.00023924956300152715]
            udict[0.9] = [0.0005733174467903793, 0.0006602686267997933, 0.0007506198471852471, 0.0008432211923890386]

            # AW0.975
            etadict[0.975] = [0.0002358678755480445, 0.0002469407185256142, 0.000263296976271812, 0.0002739185329452016]
            udict[0.975] = [0.0006574201833818061, 0.0007604279909242868, 0.0008677846142319988, 0.0009773391974974095]

        elif dts == 10:

            # AW0.375
            etadict[0.375] = [0.00027650159606670095, 0.00030210808113065367, 0.0003128818086574182, 0.0003204697848134698]
            udict[0.375] = [0.0003606791193651364, 0.0003850078726478931, 0.000397005322809595, 0.00040623395177133296]

            # AW0.4125
            etadict[0.4125] = [0.000242961475403071, 0.00026729506098841513, 0.00027665370369559287, 0.0002818599734487687]
            udict[0.4125] = [0.00032921578237872447, 0.00035424863827123197, 0.0003674728078322617, 0.00037770057535862697]

            # AW0.45
            etadict[0.45] = [0.00021703332883940763, 0.0002395957946827902, 0.0002484517598800143, 0.0002533016929161275]
            udict[0.45] = [0.00030952961345772647, 0.00033568106555571326, 0.0003519520439043436, 0.0003661401211876713]

            # AW0.4875
            etadict[0.4875] = [0.00019637218714922583, 0.00021658874484634027, 0.00022506643034514892, 0.00022988199228044794]
            udict[0.4875] = [0.0002994819567157518, 0.00032718587308590654, 0.00034743083916350363, 0.00036663304047262113]

            # AW0.525
            etadict[0.525] = [0.00017951752864751698, 0.00019711713073887824, 0.0002052554940506727, 0.0002100647076407857]
            udict[0.525] = [0.0002978318187557272, 0.0003277726437788226, 0.00035267308400260063, 0.00037746877643375724]

            # AW0.5625
            etadict[0.5625] = [0.0001659913827298096, 0.00018102544056927968, 0.00018895430663123282, 0.0001937968489493955]
            udict[0.5625] = [0.0003038702991873141, 0.0003368904417965483, 0.00036698589227653425, 0.0003977241700889247]

            # AW0.6
            etadict[0.6] = [0.00015607466793817823, 0.00016881405235011221, 0.00017675796627263747, 0.00018175241570221457]
            udict[0.6] = [0.00031698296144909904, 0.000353919751006107, 0.00038961374224722627, 0.00042650441322448444]

            # AW0.6375
            etadict[0.6375] = [0.00015032473911363934, 0.0001611630226949325, 0.00016936806467918088, 0.00017464788624983765]
            udict[0.6375] = [0.00033637747359580684, 0.0003779524287436601, 0.00041953145394541864, 0.00046270664396513513]

            # AW0.675
            etadict[0.675] = [0.00014905608970652083, 0.00015845619358329283, 0.00016712709840512326, 0.0001727877982476562]
            udict[0.675] = [0.00036107775136929443, 0.00040787892081856726, 0.0004555670734605503, 0.0005051400753713179]

            # AW0.7125
            etadict[0.7125] = [0.0001520528555598505, 0.00016050590816855224, 0.00016979112861677023, 0.00017588280415077335]
            udict[0.7125] = [0.0003900568788249539, 0.00044257208555780066, 0.0004965959822560578, 0.0005527105776756642]

            # AW0.75
            etadict[0.75] = [0.0001586421337977229, 0.00016661758696216454, 0.00017662726648699927, 0.00018317741866522526]
            udict[0.75] = [0.00042238783519978187, 0.0004810409379390843, 0.000541658464543972, 0.0006045144491721423]

            # AW0.825
            etadict[0.825] = [0.0001793125423075679, 0.00018745748927627174, 0.0001991913156732257, 0.00020671828954194]
            udict[0.825] = [0.0004943277025190604, 0.0005663962563558012, 0.0006411284638302982, 0.0007182766665292911]

            # AW0.9
            etadict[0.9] = [0.0002058456573880613, 0.0002151003021559511, 0.0002288828968591213, 0.00023747964982175784]
            udict[0.9] = [0.0005731709991120326, 0.0006600428947536359, 0.0007502994183408467, 0.0008430262917632789]

            # AW0.975
            etadict[0.975] = [0.00023560130795611053, 0.0002464280415529899, 0.00026253274949383836, 0.0002722797103620505]
            udict[0.975] = [0.0006572872760419834, 0.0007602298323566737, 0.0008675050335882126, 0.0009771970250859942]

            # AW1.05
            etadict[1.05] = [0.00026750803012079677, 0.0002801503162859563, 0.0002988152377983499, 0.0003097487310469405]
            udict[1.05] = [0.0007460151727225463, 0.0008662009770843192, 0.0009919967361278902, 0.001120058022901152]

