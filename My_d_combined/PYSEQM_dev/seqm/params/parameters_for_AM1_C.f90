      module Parameters_for_AM1_C 
      USE vast_kind_param, ONLY:  double 
!...Created by Pacific-Sierra Research 77to90  4.4G  20:31:02  03/10/06  
      real(double), dimension(107) :: ussam1, uppam1, zsam1, zpam1, zdam1, &
        betasa, betapa, alpam1, gssam1, gspam1, gppam1, gp2am1, hspam1 
      real(double), dimension(107,4) :: guesa1, guesa2, guesa3         
!                    DATA FOR ELEMENT  1       AM1:   HYDROGEN
      data ussam1(1)/  - 11.3964270D0/  
      data betasa(1)/  - 6.1737870D0/  
      data zsam1(1)/ 1.1880780D0/  
      data alpam1(1)/ 2.8823240D0/  
      data gssam1(1)/ 12.8480000D0/  
      data guesa1(1,1)/ 0.1227960D0/  
      data guesa2(1,1)/ 5.0000000D0/  
      data guesa3(1,1)/ 1.2000000D0/  
      data guesa1(1,2)/ 0.0050900D0/  
      data guesa2(1,2)/ 5.0000000D0/  
      data guesa3(1,2)/ 1.8000000D0/  
      data guesa1(1,3)/  - 0.0183360D0/  
      data guesa2(1,3)/ 2.0000000D0/  
      data guesa3(1,3)/ 2.1000000D0/  
!                    DATA FOR ELEMENT  4       AM1:   BERYLLIUM  *
      data ussam1(4)/  - 16.6023780D0/  
      data uppam1(4)/  - 10.7037710D0/  
      data betasa(4)/  - 4.0170960D0/  
      data betapa(4)/  - 4.0170960D0/  
      data zsam1(4)/ 1.0042100D0/  
      data zpam1(4)/ 1.0042100D0/  
      data alpam1(4)/ 1.6694340D0/  
      data gssam1(4)/ 9.0000000D0/  
      data gspam1(4)/ 7.4300000D0/  
      data gppam1(4)/ 6.9700000D0/  
      data gp2am1(4)/ 6.2200000D0/  
      data hspam1(4)/ 1.2800000D0/  
!                    DATA FOR ELEMENT  5       AM1:   BORON  *
      data ussam1(5)/  - 34.4928700D0/  
      data uppam1(5)/  - 22.6315250D0/  
      data betasa(5)/  - 9.5991140D0/  
      data betapa(5)/  - 6.2737570D0/  
      data zsam1(5)/ 1.6117090D0/  
      data zpam1(5)/ 1.5553850D0/  
      data alpam1(5)/ 2.4469090D0/  
      data gssam1(5)/ 10.5900000D0/  
      data gspam1(5)/ 9.5600000D0/  
      data gppam1(5)/ 8.8600000D0/  
      data gp2am1(5)/ 7.8600000D0/  
      data hspam1(5)/ 1.8100000D0/  
!                    DATA FOR ELEMENT  6       AM1:   CARBON
      data ussam1(6)/  - 52.0286580D0/  
      data uppam1(6)/  - 39.6142390D0/  
      data betasa(6)/  - 15.7157830D0/  
      data betapa(6)/  - 7.7192830D0/  
      data zsam1(6)/ 1.8086650D0/  
      data zpam1(6)/ 1.6851160D0/  
      data alpam1(6)/ 2.6482740D0/  
      data gssam1(6)/ 12.2300000D0/  
      data gspam1(6)/ 11.4700000D0/  
      data gppam1(6)/ 11.0800000D0/  
      data gp2am1(6)/ 9.8400000D0/  
      data hspam1(6)/ 2.4300000D0/  
      data guesa1(6,1)/ 0.0113550D0/  
      data guesa2(6,1)/ 5.0000000D0/  
      data guesa3(6,1)/ 1.6000000D0/  
      data guesa1(6,2)/ 0.0459240D0/  
      data guesa2(6,2)/ 5.0000000D0/  
      data guesa3(6,2)/ 1.8500000D0/  
      data guesa1(6,3)/  - 0.0200610D0/  
      data guesa2(6,3)/ 5.0000000D0/  
      data guesa3(6,3)/ 2.0500000D0/  
      data guesa1(6,4)/  - 0.0012600D0/  
      data guesa2(6,4)/ 5.0000000D0/  
      data guesa3(6,4)/ 2.6500000D0/  
!                    DATA FOR ELEMENT  7       AM1:   NITROGEN
      data ussam1(7)/  - 71.8600000D0/  
      data uppam1(7)/  - 57.1675810D0/  
      data betasa(7)/  - 20.2991100D0/  
      data betapa(7)/  - 18.2386660D0/  
      data zsam1(7)/ 2.3154100D0/  
      data zpam1(7)/ 2.1579400D0/  
      data alpam1(7)/ 2.9472860D0/  
      data gssam1(7)/ 13.5900000D0/  
      data gspam1(7)/ 12.6600000D0/  
      data gppam1(7)/ 12.9800000D0/  
      data gp2am1(7)/ 11.5900000D0/  
      data hspam1(7)/ 3.1400000D0/  
      data guesa1(7,1)/ 0.0252510D0/  
      data guesa2(7,1)/ 5.0000000D0/  
      data guesa3(7,1)/ 1.5000000D0/  
      data guesa1(7,2)/ 0.0289530D0/  
      data guesa2(7,2)/ 5.0000000D0/  
      data guesa3(7,2)/ 2.1000000D0/  
      data guesa1(7,3)/  - 0.0058060D0/  
      data guesa2(7,3)/ 2.0000000D0/  
      data guesa3(7,3)/ 2.4000000D0/  
!                    DATA FOR ELEMENT  8       AM1:   OXYGEN
      data ussam1(8)/  - 97.8300000D0/  
      data uppam1(8)/  - 78.2623800D0/  
      data betasa(8)/  - 29.2727730D0/  
      data betapa(8)/  - 29.2727730D0/  
      data zsam1(8)/ 3.1080320D0/  
      data zpam1(8)/ 2.5240390D0/  
      data alpam1(8)/ 4.4553710D0/  
      data gssam1(8)/ 15.4200000D0/  
      data gspam1(8)/ 14.4800000D0/  
      data gppam1(8)/ 14.5200000D0/  
      data gp2am1(8)/ 12.9800000D0/  
      data hspam1(8)/ 3.9400000D0/  
      data guesa1(8,1)/ 0.2809620D0/  
      data guesa2(8,1)/ 5.0000000D0/  
      data guesa3(8,1)/ 0.8479180D0/  
      data guesa1(8,2)/ 0.0814300D0/  
      data guesa2(8,2)/ 7.0000000D0/  
      data guesa3(8,2)/ 1.4450710D0/  
!                    DATA FOR ELEMENT  9       AM1:   FLUORINE  *
      data ussam1(9)/  - 136.1055790D0/  
      data uppam1(9)/  - 104.8898850D0/  
      data betasa(9)/  - 69.5902770D0/  
      data betapa(9)/  - 27.9223600D0/  
      data zsam1(9)/ 3.7700820D0/  
      data zpam1(9)/ 2.4946700D0/  
      data alpam1(9)/ 5.5178000D0/  
      data gssam1(9)/ 16.9200000D0/  
      data gspam1(9)/ 17.2500000D0/  
      data gppam1(9)/ 16.7100000D0/  
      data gp2am1(9)/ 14.9100000D0/  
      data hspam1(9)/ 4.8300000D0/  
      data guesa1(9,1)/ 0.2420790D0/  
      data guesa2(9,1)/ 4.8000000D0/  
      data guesa3(9,1)/ 0.9300000D0/  
      data guesa1(9,2)/ 0.0036070D0/  
      data guesa2(9,2)/ 4.6000000D0/  
      data guesa3(9,2)/ 1.6600000D0/  
!                    DATA FOR ELEMENT 13       AM1:   ALUMINUM  *
      data ussam1(13)/  - 24.3535850D0/  
      data uppam1(13)/  - 18.3636450D0/  
      data betasa(13)/  - 3.8668220D0/  
      data betapa(13)/  - 2.3171460D0/  
      data zsam1(13)/ 1.5165930D0/  
      data zpam1(13)/ 1.3063470D0/  
      data zdam1(13)/ 1.0000000D0/  
      data alpam1(13)/ 1.9765860D0/  
      data gssam1(13)/ 8.0900000D0/  
      data gspam1(13)/ 6.6300000D0/  
      data gppam1(13)/ 5.9800000D0/  
      data gp2am1(13)/ 5.4000000D0/  
      data hspam1(13)/ 0.7000000D0/  
      data guesa1(13,1)/ 0.0900000D0/  
      data guesa2(13,1)/ 12.3924430D0/  
      data guesa3(13,1)/ 2.0503940D0/  
!                    DATA FOR ELEMENT 14       AM1:   SILICON  *
      data ussam1(14)/  - 33.9536220D0/  
      data uppam1(14)/  - 28.9347490D0/  
      data betasa(14)/  - 3.784852D0/  
      data betapa(14)/  - 1.968123D0/  
      data zsam1(14)/ 1.830697D0/  
      data zpam1(14)/ 1.2849530D0/  
      data zdam1(14)/ 1.0000000D0/  
      data alpam1(14)/ 2.257816D0/  
      data gssam1(14)/ 9.8200000D0/  
      data gspam1(14)/ 8.3600000D0/  
      data gppam1(14)/ 7.3100000D0/  
      data gp2am1(14)/ 6.5400000D0/  
      data hspam1(14)/ 1.3200000D0/  
      data guesa1(14,1)/ 0.25D0/  
      data guesa2(14,1)/ 9.000D0/  
      data guesa3(14,1)/ 0.911453D0/  
      data guesa1(14,2)/ 0.061513D0/  
      data guesa2(14,2)/ 5.00D0/  
      data guesa3(14,2)/ 1.995569D0/  
      data guesa1(14,3)/ 0.0207890D0/  
      data guesa2(14,3)/ 5.00D0/  
      data guesa3(14,3)/ 2.990610D0/  
!                    DATA FOR ELEMENT 15        PHOSPHORUS 
      data ussam1(15)/  - 42.0298630D0/  
      data uppam1(15)/  - 34.0307090D0/  
      data betasa(15)/  - 6.3537640D0/  
      data betapa(15)/  - 6.5907090D0/  
      data zsam1(15)/ 1.9812800D0/  
      data zpam1(15)/ 1.8751500D0/  
      data zdam1(15)/ 1.0000000D0/  
      data alpam1(15)/ 2.4553220D0/  
      data gssam1(15)/ 11.5600050D0/  
      data gspam1(15)/ 5.2374490D0/  
      data gppam1(15)/ 7.8775890D0/  
      data gp2am1(15)/ 7.3076480D0/  
      data hspam1(15)/ 0.7792380D0/  
      data guesa1(15,1)/  - 0.0318270D0/  
      data guesa2(15,1)/ 6.0000000D0/  
      data guesa3(15,1)/ 1.4743230D0/  
      data guesa1(15,2)/ 0.0184700D0/  
      data guesa2(15,2)/ 7.0000000D0/  
      data guesa3(15,2)/ 1.7793540D0/  
      data guesa1(15,3)/ 0.0332900D0/  
      data guesa2(15,3)/ 9.0000000D0/  
      data guesa3(15,3)/ 3.0065760D0/  
!                    DATA FOR ELEMENT 16       AM1:   SULFUR  *
!
      data ussam1(16)/  - 56.6940560D0/  
      data uppam1(16)/  - 48.7170490D0/  
      data betasa(16)/  - 3.9205660D0/  
      data betapa(16)/  - 7.9052780D0/  
      data zsam1(16)/ 2.3665150D0/  
      data zpam1(16)/ 1.6672630D0/  
      data zdam1(16)/ 1.0000000D0/  
      data alpam1(16)/ 2.4616480D0/  
      data gssam1(16)/ 11.7863290D0/  
      data gspam1(16)/ 8.6631270D0/  
      data gppam1(16)/ 10.0393080D0/  
      data gp2am1(16)/ 7.7816880D0/  
      data hspam1(16)/ 2.5321370D0/  
      data guesa1(16,1)/  - 0.5091950D0/  
      data guesa2(16,1)/ 4.5936910D0/  
      data guesa3(16,1)/ 0.7706650D0/  
      data guesa1(16,2)/  - 0.0118630D0/  
      data guesa2(16,2)/ 5.8657310D0/  
      data guesa3(16,2)/ 1.5033130D0/  
      data guesa1(16,3)/ 0.0123340D0/  
      data guesa2(16,3)/ 13.5573360D0/  
      data guesa3(16,3)/ 2.0091730D0/  
!                    DATA FOR ELEMENT 17       AM1:   CHLORINE  *
      data ussam1(17)/  - 111.6139480D0/  
      data uppam1(17)/  - 76.6401070D0/  
      data betasa(17)/  - 24.5946700D0/  
      data betapa(17)/  - 14.6372160D0/  
      data zsam1(17)/ 3.6313760D0/  
      data zpam1(17)/ 2.0767990D0/  
      data zdam1(17)/ 1.0000000D0/  
      data alpam1(17)/ 2.9193680D0/  
      data gssam1(17)/ 15.0300000D0/  
      data gspam1(17)/ 13.1600000D0/  
      data gppam1(17)/ 11.3000000D0/  
      data gp2am1(17)/ 9.9700000D0/  
      data hspam1(17)/ 2.4200000D0/  
      data guesa1(17,1)/ 0.0942430D0/  
      data guesa2(17,1)/ 4.0000000D0/  
      data guesa3(17,1)/ 1.3000000D0/  
      data guesa1(17,2)/ 0.0271680D0/  
      data guesa2(17,2)/ 4.0000000D0/  
      data guesa3(17,2)/ 2.1000000D0/  
!                    DATA FOR ELEMENT 30        ZINC
      data ussam1(30)/  - 21.0400080D0/  
      data uppam1(30)/  - 17.6555740D0/  
      data betasa(30)/  - 1.9974290D0/  
      data betapa(30)/  - 4.7581190D0/  
      data zsam1(30)/ 1.9542990D0/  
      data zpam1(30)/ 1.3723650D0/  
      data zdam1(30)/ 1.0000000D0/  
      data alpam1(30)/ 1.4845630D0/  
      data gssam1(30)/ 11.8000000D0/  
      data gspam1(30)/ 11.1820180D0/  
      data gppam1(30)/ 13.3000000D0/  
      data gp2am1(30)/ 12.9305200D0/  
      data hspam1(30)/ 0.4846060D0/  
!                    DATA FOR ELEMENT 32        GERMANIUM
      data ussam1(32)/  - 34.1838890D0/  
      data uppam1(32)/  - 28.6408110D0/  
      data betasa(32)/  - 4.3566070D0/  
      data betapa(32)/  - 0.9910910D0/  
      data zsam1(32)/ 1.2196310D0/  
      data zpam1(32)/ 1.9827940D0/  
      data alpam1(32)/ 2.1364050D0/  
      data gssam1(32)/ 10.1686050D0/  
      data gspam1(32)/ 8.1444730D0/  
      data gppam1(32)/ 6.6719020D0/  
      data gp2am1(32)/ 6.2697060D0/  
      data hspam1(32)/ 0.9370930D0/  
!                    DATA FOR ELEMENT 33      ARSENIC 
      data ussam1(33)/  - 41.6817510D0/  
      data uppam1(33)/  - 33.4506152D0/  
      data betasa(33)/  - 5.6481504D0/  
      data betapa(33)/  - 4.9979109D0/  
      data zsam1(33)/ 2.2576897D0/  
      data zpam1(33)/ 1.7249710D0/  
      data alpam1(33)/ 2.2405380D0/  
      data gssam1(33)/ 11.0962258D0/  
      data gspam1(33)/ 4.9259328D0/  
      data gppam1(33)/ 7.8781648D0/  
      data gp2am1(33)/ 7.5961088D0/  
      data hspam1(33)/ 0.6246173D0/  
      data guesa1(33,1)/  - 0.0073614D0/  
      data guesa2(33,1)/ 4.9433993D0/  
      data guesa3(33,1)/ 1.4544264D0/  
      data guesa1(33,2)/ 0.0437629D0/  
      data guesa2(33,2)/ 3.1944613D0/  
      data guesa3(33,2)/ 2.0144939D0/  
!                    DATA FOR ELEMENT 34        SELENIUM
      data ussam1(34)/  - 41.9984056D0/  
      data uppam1(34)/  - 32.8575485D0/  
      data betasa(34)/  - 3.1470826D0/  
      data betapa(34)/  - 6.1468406D0/  
      data zsam1(34)/ 2.6841570D0/  
      data zpam1(34)/ 2.0506164D0/  
      data alpam1(34)/ 2.6375694D0/  
      data gssam1(34)/ 6.7908891D0/  
      data gspam1(34)/ 6.4812786D0/  
      data gppam1(34)/ 6.4769273D0/  
      data gp2am1(34)/ 5.2796993D0/  
      data hspam1(34)/ 4.4548356D0/  
      data guesa1(34,1)/ 0.1116681D0/  
      data guesa2(34,1)/ 6.5086644D0/  
      data guesa3(34,1)/ 1.4981077D0/  
      data guesa1(34,2)/ 0.0396143D0/  
      data guesa2(34,2)/ 6.5241228D0/  
      data guesa3(34,2)/ 2.0751916D0/  
!                    DATA FOR ELEMENT 35       AM1:   BROMINE  * 
      data ussam1(35)/  - 104.6560630D0/  
      data uppam1(35)/  - 74.9300520D0/  
      data betasa(35)/  - 19.3998800D0/  
      data betapa(35)/  - 8.9571950D0/  
      data zsam1(35)/ 3.0641330D0/  
      data zpam1(35)/ 2.0383330D0/  
      data zdam1(35)/ 1.0000000D0/  
      data alpam1(35)/ 2.5765460D0/  
      data gssam1(35)/ 15.0364395D0/  
      data gspam1(35)/ 13.0346824D0/  
      data gppam1(35)/ 11.2763254D0/  
      data gp2am1(35)/ 9.8544255D0/  
      data hspam1(35)/ 2.4558683D0/  
      data guesa1(35,1)/ 0.0666850D0/  
      data guesa2(35,1)/ 4.0000000D0/  
      data guesa3(35,1)/ 1.5000000D0/  
      data guesa1(35,2)/ 0.0255680D0/  
      data guesa2(35,2)/ 4.0000000D0/  
      data guesa3(35,2)/ 2.3000000D0/  
!                    DATA FOR ELEMENT 51        ANTIMONY
      data ussam1(51)/  - 44.4381620D0/  
      data uppam1(51)/  - 32.3895140D0/  
      data betasa(51)/  - 7.3823300D0/  
      data betapa(51)/  - 3.6331190D0/  
      data zsam1(51)/ 2.2548230D0/  
      data zpam1(51)/ 2.2185920D0/  
      data alpam1(51)/ 2.2763310D0/  
      data gssam1(51)/ 11.4302510D0/  
      data gspam1(51)/ 5.7879220D0/  
      data gppam1(51)/ 6.4240940D0/  
      data gp2am1(51)/ 6.8491810D0/  
      data hspam1(51)/ 0.5883400D0/  
      data guesa1(51,1)/  - 0.5964470D0/  
      data guesa2(51,1)/ 6.0279500D0/  
      data guesa3(51,1)/ 1.7103670D0/  
      data guesa1(51,2)/ 0.8955130D0/  
      data guesa2(51,2)/ 3.0281090D0/  
      data guesa3(51,2)/ 1.5383180D0/  
!                    DATA FOR ELEMENT 52        TELLURIUM
      data ussam1(52)/  - 39.2454230D0/  
      data uppam1(52)/  - 30.8515845D0/  
      data betasa(52)/  - 8.3897294D0/  
      data betapa(52)/  - 5.1065429D0/  
      data zsam1(52)/ 2.1321165D0/  
      data zpam1(52)/ 1.9712680D0/  
      data alpam1(52)/ 6.0171167D0/  
      data gssam1(52)/ 4.9925231D0/  
      data gspam1(52)/ 4.9721484D0/  
      data gppam1(52)/ 7.2097852D0/  
      data gp2am1(52)/ 5.6211521D0/  
      data hspam1(52)/ 4.0071821D0/  
      data guesa1(52,1)/ 0.4873378D0/  
      data guesa2(52,1)/ 6.0519413D0/  
      data guesa3(52,1)/ 1.3079857D0/  
      data guesa1(52,2)/ 0.1520464D0/  
      data guesa2(52,2)/ 3.8304067D0/  
      data guesa3(52,2)/ 2.0899707D0/  
!                    DATA FOR ELEMENT 53       AM1:   IODINE  *
      data ussam1(53)/  - 103.5896630D0/  
      data uppam1(53)/  - 74.4299970D0/  
      data betasa(53)/  - 8.4433270D0/  
      data betapa(53)/  - 6.3234050D0/  
      data zsam1(53)/ 2.1028580D0/  
      data zpam1(53)/ 2.1611530D0/  
      data zdam1(53)/ 1.0000000D0/  
      data alpam1(53)/ 2.2994240D0/  
      data gssam1(53)/ 15.0404486D0/  
      data gspam1(53)/ 13.0565580D0/  
      data gppam1(53)/ 11.1477837D0/  
      data gp2am1(53)/ 9.9140907D0/  
      data hspam1(53)/ 2.4563820D0/  
      data guesa1(53,1)/ 0.0043610D0/  
      data guesa2(53,1)/ 2.3000000D0/  
      data guesa3(53,1)/ 1.8000000D0/  
      data guesa1(53,2)/ 0.0157060D0/  
      data guesa2(53,2)/ 3.0000000D0/  
      data guesa3(53,2)/ 2.2400000D0/  
!
!       Data for Element  57:     Lanthanum
!    
      data  alpam1( 57)/       2.1879021346d0/
      data gssam1(57) /       55.7344864002d0/
      data guesa1( 57,1)/       1.3207809006d0/
      data guesa2( 57,1)/       7.1394307023d0/
      data guesa3( 57,1)/       1.8503281529d0/
      data guesa1( 57,2)/       0.3425777564d0/
      data guesa2( 57,2)/       8.7780631664d0/
      data guesa3( 57,2)/       3.1678964355d0/
!
!       Data for Element  58:     Cerium
!   
      data     alpam1( 58)  /       2.6637769616d0/
      data      gssam1(58)  /       58.7223887052d0/
      data   guesa1( 58,1)/       1.7507655141d0/
      data   guesa2( 58,1)/       7.6163181355d0/
      data   guesa3( 58,1)/       1.8064852538d0/
      data   guesa1( 58,2)/       0.0093401239d0/
      data   guesa2( 58,2)/       8.7664931283d0/
      data   guesa3( 58,2)/       3.2008171269d0/
!
!       Data for Element  59:     Praseodymium
!
      data     alpam1( 59)  /       2.6104229985d0/
      data      gssam1(59)  /       58.9017644267d0/
      data   guesa1( 59,1)/       1.7515391427d0/
      data   guesa2( 59,1)/       7.6039742620d0/
      data   guesa3( 59,1)/       1.8084677103d0/
      data   guesa1( 59,2)/       0.0097057032d0/
      data   guesa2( 59,2)/       8.7264195205d0/
      data   guesa3( 59,2)/       2.9111890014d0/
!
!       Data for Element  60:     Neodymium
!    
      data     alpam1( 60)  /       4.5002951307d0/
      data      gssam1(60)  /       57.6242766015d0/
      data   guesa1( 60,1)/       1.1206946050d0/
      data   guesa2( 60,1)/       6.8295606379d0/
      data   guesa3( 60,1)/       1.7859049866d0/
      data   guesa1( 60,2)/       0.1070369350d0/
      data   guesa2( 60,2)/       10.7894804795d0/
      data   guesa3( 60,2)/       3.1628661485d0/
!
!       Data for Element  61:     Promethium
!   
      data     alpam1( 61)  /       3.1059833647d0/
      data      gssam1(61)  /       59.4249705519d0/
      data   guesa1( 61,1)/       1.7347671158d0/
      data   guesa2( 61,1)/       9.2464226360d0/
      data   guesa3( 61,1)/       1.7533419485d0/
      data   guesa1( 61,2)/       0.2571017258d0/
      data   guesa2( 61,2)/       7.8793445267d0/
      data   guesa3( 61,2)/       3.0498162940d0/
!
!       Data for Element  62:     Samarium
!   
      data     alpam1( 62)/       4.1758509d0/
      data      gssam1(62)  /       56.9935144820d0/
      data guesa1( 62,1)/       0.9592885d0/
      data guesa2( 62,1)/       6.4799924d0/
      data guesa3( 62,1)/       1.7381402d0/
      data guesa1( 62,2)/       0.0261004d0/
      data guesa2( 62,2)/       9.7391952d0/
      data guesa3( 62,2)/       2.8881177d0/
!
!       Data for Element  63:     Europium
!   
      data     alpam1( 63)/       2.1247189d0/
      data gssam1(63) /       55.6059122033D0/
      data guesa1( 63,1)/       0.5695122d0/
      data guesa2( 63,1)/       7.4680208d0/
      data guesa3( 63,1)/       1.7319730d0/
      data guesa1( 63,2)/       0.3286619d0/
      data guesa2( 63,2)/       7.8009780d0/
      data guesa3( 63,2)/       2.9641285d0/
!
!       Data for Element  64:   Gadolinium
!         
      data     alpam1( 64)/       3.6525485d0/
      data    gssam1(64) /       55.7083247618D0/
      data guesa1( 64,1)/       0.7013512d0/
      data guesa2( 64,1)/       7.5454483d0/
      data guesa3( 64,1)/       1.7761953d0/
      data guesa1( 64,2)/       0.1293094d0/
      data guesa2( 64,2)/       8.3437991d0/
      data guesa3( 64,2)/       3.0110320d0/
!
!       Data for Element  65:      Terbium
!       
      data     alpam1( 65)/       2.3418889d0/
      data    gssam1(65) /       55.7245956904D0/
      data guesa1( 65,1)/       0.7734458d0/
      data guesa2( 65,1)/       7.6510526d0/
      data guesa3( 65,1)/       1.7033464d0/
      data guesa1( 65,2)/       0.3936233d0/
      data guesa2( 65,2)/       7.9261457d0/
      data guesa3( 65,2)/       3.0132951d0/
!
!       Data for Element  66:     Dysprosium
!
      data     alpam1( 66)  /       2.4164925319d0/
      data      gssam1(66)  /       55.7676495431d0/
      data   guesa1( 66,1)/       1.0385214022d0/
      data   guesa2( 66,1)/       8.0016176963d0/
      data   guesa3( 66,1)/       1.7161371716d0/
      data   guesa1( 66,2)/       0.3018081665d0/
      data   guesa2( 66,2)/       8.7917318591d0/
      data   guesa3( 66,2)/       2.9953820978d0/
!
!       Data for Element  67:     Holmium
!  
      data     alpam1( 67)  /       3.5558255806d0/
      data      gssam1(67)  /       58.0375569983d0/
      data   guesa1( 67,1)/       0.9819487665d0/
      data   guesa2( 67,1)/       8.9158636523d0/
      data   guesa3( 67,1)/       1.7513929352d0/
      data   guesa1( 67,2)/       0.5454296933d0/
      data   guesa2( 67,2)/       8.6358143694d0/
      data   guesa3( 67,2)/       3.0176828845d0/
!
!       Data for Element  68:     Erbium
!    
      data     alpam1( 68)  /       3.6568232540d0/
      data      gssam1(68)  /       58.0489423317d0/
      data   guesa1( 68,1)/       0.7029401580d0/
      data   guesa2( 68,1)/       8.7235009642d0/
      data   guesa3( 68,1)/       1.7746084736d0/
      data   guesa1( 68,2)/       0.1321261631d0/
      data   guesa2( 68,2)/       8.3498075890d0/
      data   guesa3( 68,2)/       3.0114806813d0/
!
!       Data for Element  69:     Thulium
! 
      data     alpam1( 69)  /       2.6969696589d0/
      data      gssam1(69)  /       55.8847434446d0/
      data   guesa1( 69,1)/       0.9252928725d0/
      data   guesa2( 69,1)/       7.8636142242d0/
      data   guesa3( 69,1)/       1.7113099330d0/
      data   guesa1( 69,2)/       0.4125805849d0/
      data   guesa2( 69,2)/       8.5211941024d0/
      data   guesa3( 69,2)/       2.9817874911d0/
!
!       Data for Element  70:     Ytterbium
!
      data     alpam1( 70)  /       4.0022390936d0/
      data      gssam1(70)  /       56.1788359906d0/
      data   guesa1( 70,1)/       1.0231210543d0/
      data   guesa2( 70,1)/       8.3969197698d0/
      data   guesa3( 70,1)/       1.7046767332d0/
      data   guesa1( 70,2)/       0.3351965495d0/
      data   guesa2( 70,2)/       7.2817207497d0/
      data   guesa3( 70,2)/       2.9140122029d0/
!
!       Data for Element  71:     Lutetium
!   
      data     alpam1( 71)  /       4.0203424467d0/
      data      gssam1(71)  /       56.1751741742d0/
      data   guesa1( 71,1)/       1.0381638761d0/
      data   guesa2( 71,1)/       8.4911797242d0/
      data   guesa3( 71,1)/       1.7034420896d0/
      data   guesa1( 71,2)/       0.3342233253d0/
      data   guesa2( 71,2)/       7.2729946604d0/
      data   guesa3( 71,2)/       2.9153096100d0/

!                    DATA FOR ELEMENT 80        MERCURY
      data ussam1(80)/  - 19.9415780D0/  
      data uppam1(80)/  - 11.1108700D0/  
      data betasa(80)/  - 0.9086570D0/  
      data betapa(80)/  - 4.9093840D0/  
      data zsam1(80)/ 2.0364130D0/  
      data zpam1(80)/ 1.9557660D0/  
      data alpam1(80)/ 1.4847340D0/  
      data gssam1(80)/ 10.8000000D0/  
      data gspam1(80)/ 9.3000000D0/  
      data gppam1(80)/ 14.3000000D0/  
      data gp2am1(80)/ 13.5000000D0/  
      data hspam1(80)/ 1.3000000D0/  

   end module Parameters_for_AM1_C 
