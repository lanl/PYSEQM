eV

BIASES for Etot
Non-D3H4
ONCH: -1755.6180, N:-1310.7279,  C:-914.8454,    H:-2.6524

With D3H4
ONCH: -1755.6142, -1310.7154,  -914.8211,    -2.6722


BIASES fro Eatomiz
Non-D3H4
ONCH: -0.0548, -0.2329, -0.0251, -0.2406

With D3H4
ONCH: -0.0392, -0.2209, -0.0151, -0.2526

Z - atomic numbers
R - atomic coordinates
Q - atomic charges


Etot_DFT - DFT energies
Etot_PM6 - PM6 total energies (without D3H4 correction)
Et_m_bias - total DFT energies with biases subtracted to adjust to PM6 energy scale (use for training together with F_DFT)
F_DFT - DFT forces (use for training together with Et_m_bias)

E_D3 - D3 energy corrections to PM6
E_H4 - H4 energy corrections to PM6
F_D3 - D3 force corrections to PM6
F_H4 - H4 force corrections to PM6

Et_m_d3h4bias - total DFT energies with biases subtracted to adjust to PM6-D3H4 energy scale (use for training together with F_DFT_m_D3H4)
F_DFT_m_D3H4 - DFT forces with D3H4 corrections for PM6 subtracted (use for training to Et_m_d3h4bias) 

Eatomiz_PM6 - PM6 atomization energies (without D3H4 correction)
F_PM6 - PM6 forces (without D3H4 correction)












