      module Parameters_for_MNDO_C 
      USE vast_kind_param, ONLY:  double 
!...Created by Pacific-Sierra Research 77to90  4.4G  20:31:02  03/10/06  
      real(double), dimension(107) :: ussm, uppm, uddm, zsm, zpm, zdm, betasm, &
        betapm, alpm, gssm, gspm, gppm, gp2m, hspm, polvom 

 !                    DATA FOR ELEMENT  1        HYDROGEN
   data alpm(1)/2.5441341d0/
   data betasm(1)/-6.9890640d0/
   data gssm(1)/12.848d00/
   data polvom(1)/0.2287d0/
   data ussm(1)/-11.9062760d0/
   data zsm(1)/1.3319670d0/
!                    DATA FOR ELEMENT  3        LITHIUM
   data alpm(3)/1.2501400d0/
   data betapm(3)/-1.3500400d0/
   data betasm(3)/-1.3500400d0/
   data gp2m(3)/4.5200000d0/
   data gppm(3)/5.0000000d0/
   data gspm(3)/5.4200000d0/
   data gssm(3)/7.3000000d0/
   data hspm(3)/0.8300000d0/
   data uppm(3)/-2.7212000d0/
   data ussm(3)/-5.1280000d0/
   data zpm(3)/0.7023800d0/
   data zsm(3)/0.7023800d0/
!                    DATA FOR ELEMENT  4        BERYLLIUM
   data alpm(4)/1.6694340d0/
   data betapm(4)/-4.0170960d0/
   data betasm(4)/-4.0170960d0/
   data gp2m(4)/6.22d00/
   data gppm(4)/6.97d00/
   data gspm(4)/7.43d00/
   data gssm(4)/9.00d00/
   data hspm(4)/1.28d00/
   data uppm(4)/-10.7037710d0/
   data ussm(4)/-16.6023780d0/
   data zpm(4)/1.0042100d0/
   data zsm(4)/1.0042100d0/
!                    DATA FOR ELEMENT  5        BORON
   data alpm(5)/2.1349930d0/
   data betapm(5)/-8.2520540d0/
   data betasm(5)/-8.2520540d0/
   data gp2m(5)/7.86d00/
   data gppm(5)/8.86d00/
   data gspm(5)/9.56d00/
   data gssm(5)/10.59d00/
   data hspm(5)/1.81d00/
   data uppm(5)/-23.1216900d0/
   data ussm(5)/-34.5471300d0/
   data zpm(5)/1.5068010d0/
   data zsm(5)/1.5068010d0/
!                    DATA FOR ELEMENT  6        CARBON
   data alpm(6)/2.5463800d0/
   data betapm(6)/-7.9341220d0/
   data betasm(6)/-18.9850440d0/
   data gp2m(6)/9.84d00/
   data gppm(6)/11.08d00/
   data gspm(6)/11.47d00/
   data gssm(6)/12.23d00/
   data hspm(6)/2.43d00/
   data polvom(6)/0.2647d0/
   data uppm(6)/-39.2055580d0/
   data ussm(6)/-52.2797450d0/
   data zpm(6)/1.7875370d0/
   data zsm(6)/1.7875370d0/
!                    DATA FOR ELEMENT  7        NITROGEN
   data alpm(7)/2.8613420d0/
   data betapm(7)/-20.4957580d0/
   data betasm(7)/-20.4957580d0/
   data gp2m(7)/11.59d00/
   data gppm(7)/12.98d00/
   data gspm(7)/12.66d00/
   data gssm(7)/13.59d00/
   data hspm(7)/3.14d00/
   data polvom(7)/0.3584d0/
   data uppm(7)/-57.1723190d0/
   data ussm(7)/-71.9321220d0/
   data zpm(7)/2.2556140d0/
   data zsm(7)/2.2556140d0/
!                    DATA FOR ELEMENT  8        OXYGEN
   data alpm(8)/3.1606040d0/
   data betapm(8)/-32.6880820d0/
   data betasm(8)/-32.6880820d0/
   data gp2m(8)/12.98d00/
   data gppm(8)/14.52d00/
   data gspm(8)/14.48d00/
   data gssm(8)/15.42d00/
   data hspm(8)/3.94d00/
   data polvom(8)/0.2324d0/
   data uppm(8)/-77.7974720d0/
   data ussm(8)/-99.6443090d0/
   data zpm(8)/2.6999050d0/
   data zsm(8)/2.6999050d0/
   data alpm(9)/3.4196606d0/
!                    DATA FOR ELEMENT  9        FLUORINE
   data betapm(9)/-36.5085400d0/
   data betasm(9)/-48.2904660d0/
   data gp2m(9)/14.91d00/
   data gppm(9)/16.71d00/
   data gspm(9)/17.25d00/
   data gssm(9)/16.92d00/
   data hspm(9)/4.83d00/
   data polvom(9)/0.1982d0/
   data uppm(9)/-105.7821370d0/
   data ussm(9)/-131.0715480d0/
   data zpm(9)/2.8484870d0/
   data zsm(9)/2.8484870d0/
!                    DATA FOR ELEMENT  11        SODIUM
   data alpm(11)/1.2619400d0/
   data betapm(11)/-1.1410100d0/
   data betasm(11)/-1.1452900d0/
   data gp2m(11)/4.2330000d0/
   data gppm(11)/4.2330000d0/
   data gspm(11)/5.1800000d0/
   data gssm(11)/7.2000000d0/
   data hspm(11)/0.6428000d0/
   data uppm(11)/-2.3643000d0/
   data ussm(11)/-4.6110000d0/
   data zpm(11)/0.8295700d0/
   data zsm(11)/0.7338400d0/
!                    DATA FOR ELEMENT 13        ALUMINUM
   data alpm(13)/1.8688394d0/
   data betapm(13)/-2.6702840d0/
   data betasm(13)/-2.6702840d0/
   data gp2m(13)/5.40d00/
   data gppm(13)/5.98d00/
   data gspm(13)/6.63d00/
   data gssm(13)/8.09d00/
   data hspm(13)/0.70d00/
   data uppm(13)/-17.5198780d0/
   data ussm(13)/-23.8070970d0/
   data zdm(13)/1.0000000d0/
   data zpm(13)/1.4441610d0/
   data zsm(13)/1.4441610d0/
!                    DATA FOR ELEMENT 14          SILICON
   data alpm(14)/2.2053160d0/
   data betapm(14)/-1.0758270d0/
   data betasm(14)/-9.0868040d0/
   data gp2m(14)/6.54d00/
   data gppm(14)/7.31d00/
   data gspm(14)/8.36d00/
   data gssm(14)/9.82d00/
   data hspm(14)/1.32d00/
   data uppm(14)/-27.7696780d0/
   data ussm(14)/-37.0375330d0/
   data zdm(14)/1.0000000d0/
   data zpm(14)/1.7099430d0/
   data zsm(14)/1.3159860d0/
!                    DATA FOR ELEMENT 15        PHOSPHORUS
   data alpm(15)/2.4152800d0/
   data betapm(15)/-6.7916000d0/
   data betasm(15)/-6.7916000d0/
   data gp2m(15)/7.68d00/
   data gppm(15)/8.64d00/
   data gspm(15)/10.08d00/
   data gssm(15)/11.56d00/
   data hspm(15)/1.92d00/
   data uppm(15)/-42.8510800d0/
   data ussm(15)/-56.1433600d0/
   data zdm(15)/1.0000000d0/
   data zpm(15)/1.7858100d0/
   data zsm(15)/2.1087200d0/
!                    DATA FOR ELEMENT 16        SULFUR
   data alpm(16)/2.4780260d0/
   data betapm(16)/-10.1084330d0/
   data betasm(16)/-10.7616700d0/
   data gp2m(16)/8.83d00/
   data gppm(16)/9.90d00/
   data gspm(16)/11.26d00/
   data gssm(16)/12.88d00/
   data hspm(16)/2.26d00/
   data uppm(16)/-56.9732070d0/
   data ussm(16)/-72.2422810d0/
   data zdm(16)/1.0000000d0/
   data zpm(16)/2.0091460d0/
   data zsm(16)/2.3129620d0/
!                    DATA FOR ELEMENT 17        CHLORINE
   data alpm(17)/2.5422010d0/
   data betapm(17)/-14.2623200d0/
   data betasm(17)/-14.2623200d0/
   data gp2m(17)/9.97d00/
   data gppm(17)/11.30d00/
   data gspm(17)/13.16d00/
   data gssm(17)/15.03d00/
   data hspm(17)/2.42d00/
   data polvom(17)/1.3236d0/
   data uppm(17)/-77.3786670d0/
   data ussm(17)/-100.2271660d0/
   data zdm(17)/1.0000000d0/
   data zpm(17)/2.0362630d0/
   data zsm(17)/3.7846450d0/
!                    DATA FOR ELEMENT  19        POTASSIUM
   data alpm(19)/1.2737000d0/
   data betapm(19)/-0.9320000d0/
   data betasm(19)/-0.9405000d0/
   data gp2m(19)/3.9468000d0/
   data gppm(19)/5.8638000d0/
   data gspm(19)/4.9411000d0/
   data gssm(19)/7.1030000d0/
   data hspm(19)/0.4557000d0/
   data uppm(19)/-2.0075000d0/
   data ussm(19)/-4.0934000d0/
   data zpm(19)/0.9568000d0/
   data zsm(19)/0.7653000d0/
!                    DATA FOR ELEMENT 30        ZINC
   data alpm(30)/1.5064570d0/
   data betapm(30)/-2.0000000d0/
   data betasm(30)/-1.0000000d0/
   data gp2m(30)/12.9305200d0/
   data gppm(30)/13.3000000d0/
   data gspm(30)/11.1820180d0/
   data gssm(30)/11.8000000d0/
   data hspm(30)/0.4846060d0/
   data uppm(30)/-19.6252240d0/
   data ussm(30)/-20.8397160d0/
   data zdm(30)/1.0000000d0/
   data zpm(30)/1.4609460d0/
   data zsm(30)/2.0473590d0/
!                    DATA FOR ELEMENT 32        GERMANIUM
   data alpm(32)/1.9784980d0/
   data betapm(32)/-1.7555170d0/
   data betasm(32)/-4.5164790d0/
   data gp2m(32)/6.5000000d0/
   data gppm(32)/7.3000000d0/
   data gspm(32)/8.3000000d0/
   data gssm(32)/9.8000000d0/
   data hspm(32)/1.3000000d0/
   data uppm(32)/-27.4251050d0/
   data ussm(32)/-33.9493670d0/
   data zpm(32)/2.0205640d0/
   data zsm(32)/1.2931800d0/
!                    DATA FOR ELEMENT 35        BROMINE
   data alpm(35)/2.4457051d0/
   data betapm(35)/-9.9437400d0/
   data betasm(35)/-8.9171070d0/
   data gp2m(35)/9.85442552d0/
   data gppm(35)/11.27632539d0/
   data gspm(35)/13.03468242d0/
   data gssm(35)/15.03643948d0/
   data hspm(35)/2.45586832d0/
   data polvom(35)/2.2583d0/
   data uppm(35)/-75.6713075d0/
   data ussm(35)/-99.9864405d0/
   data zdm(35)/1.0000000d0/
   data zpm(35)/2.1992091d0/
   data zsm(35)/3.8543019d0/
!                    DATA FOR ELEMENT 50        TIN
   data alpm(50)/1.8008140d0/
   data betapm(50)/-4.2904160d0/
   data betasm(50)/-3.2351470d0/
   data gp2m(50)/6.5000000d0/
   data gppm(50)/7.3000000d0/
   data gspm(50)/8.3000000d0/
   data gssm(50)/9.8000000d0/
   data hspm(50)/1.3000000d0/
   data uppm(50)/-28.5602490d0/
   data ussm(50)/-40.8518020d0/
   data zpm(50)/1.9371060d0/
   data zsm(50)/2.0803800d0/
!                    DATA FOR ELEMENT 53        IODINE
   data alpm(53)/2.2073200d0/
   data betapm(53)/-6.1967810d0/
   data betasm(53)/-7.4144510d0/
   data gp2m(53)/9.91409071d0/
   data gppm(53)/11.14778369d0/
   data gspm(53)/13.05655798d0/
   data gssm(53)/15.04044855d0/
   data hspm(53)/2.45638202d0/
   data polvom(53)/4.0930d0/
   data uppm(53)/-74.6114692d0/
   data ussm(53)/-100.0030538d0/
   data zdm(53)/1.0000000d0/
   data zpm(53)/2.1694980d0/
   data zsm(53)/2.2729610d0/
!                    DATA FOR ELEMENT 80        MERCURY
   data alpm(80)/1.3356410d0/
   data betapm(80)/-6.2066830d0/
   data betasm(80)/-0.4045250d0/
   data gp2m(80)/13.5000000d0/
   data gppm(80)/14.3000000d0/
   data gspm(80)/9.3000000d0/
   data gssm(80)/10.8000000d0/
   data hspm(80)/1.3000000d0/
   data uppm(80)/-13.1025300d0/
   data ussm(80)/-19.8095740d0/
   data zpm(80)/2.0650380d0/
   data zsm(80)/2.2181840d0/
!                    DATA FOR ELEMENT 82        LEAD
   data alpm(82)/1.7283330d0/
   data betapm(82)/-3.0000000d0/
   data betasm(82)/-8.0423870d0/
   data gp2m(82)/6.5000000d0/
   data gppm(82)/7.3000000d0/
   data gspm(82)/8.3000000d0/
   data gssm(82)/9.8000000d0/
   data hspm(82)/1.3000000d0/
   data uppm(82)/-28.8475600d0/
   data ussm(82)/-47.3196920d0/
   data zpm(82)/2.0820710d0/
   data zsm(82)/2.4982860d0/
!
!     START OF "OLD" ELEMENTS: THESE ARE OLD PARAMETERS WHICH
!     CAN BE USED, IF DESIRED, BY SPECIFYING "<CHEMICAL SYMBOL>YEAR"
!     AS IN SI1978 OR  S1983.
!
!                    DATA FOR ELEMENT 90        SILICON
   data alpm(90)/2.1961078d0/
   data betapm(90)/-4.2562180d0/
   data betasm(90)/-4.2562180d0/
   data gp2m(90)/6.54d00/
   data gppm(90)/7.31d00/
   data gspm(90)/8.36d00/
   data gssm(90)/9.82d00/
   data hspm(90)/1.32d00/
   data uppm(90)/-28.0891870d0/
   data ussm(90)/-40.5682920d0/
   data zdm(90)/1.0000000d0/
   data zpm(90)/1.4353060d0/
   data zsm(90)/1.4353060d0/
! S ??
   data alpm(91)/2.4916445d0/
   data betapm(91)/-11.1422310d0/
   data betasm(91)/-11.1422310d0/
   data uppm(91)/-57.8320130d0/
   data ussm(91)/-75.2391520d0/
   data zdm(91)/1.0000000d0/
   data zpm(91)/2.0343930d0/
   data zsm(91)/2.6135910d0/
! Capped Bond
   data alpm(102)/2.5441341d0/
   data betasm(102)/-9999999.0000000d0/
   data gssm(102)/12.8480000d0/
   data hspm(102)/0.1000000d0/
   data ussm(102)/-11.9062760d0/
   data zdm(102)/0.3000000d0/
   data zpm(102)/0.3000000d0/
   data zsm(102)/4.0000000d0/
!                               DATA FOR THE " ++ " SPARKLE
   data alpm(103)/1.5d0/
!                               DATA FOR THE " + " SPARKLE
   data alpm(104)/1.5d0/
!                               DATA FOR THE " -- " SPARKLE
   data alpm(105)/1.5d0/
!                               DATA FOR THE " - " SPARKLE
   data alpm(106)/1.5d0/
      end module Parameters_for_MNDO_C 
