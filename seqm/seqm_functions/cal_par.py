import torch
from .constants import ev


#dd = lambda qn, zs, zp : (2.0*qn+1.0)*(4.0*zs*zp)**(qn+0.5)/(zs+zp)**(2.0*qn+2)/np.sqrt(3.0)
#qq = lambda qn, zp : np.sqrt((4.0*qn**2+6.0*qn+2.0)/20.0)/zp

def dd_qq(qn, zs, zp):
    """
    qn: pricipal quantum number for valence shell, shape (n_atoms,)
    zs : zeta_s, shape (n_atoms,)
    zp : zeta_p, shape (n_atoms,)
    return dd, qq
      dd: dipole charge separation
      qq: qutrupole charge separation
    """
    dd = (2.0*qn+1.0)*(4.0*zs*zp)**(qn+0.5)/(zs+zp)**(2.0*qn+2.0)/ \
           torch.sqrt(torch.tensor(3.0, dtype=zs.dtype, device=zs.device))
    qq = torch.sqrt((4.0*qn**2+6.0*qn+2.0)/20.0)/zp
    return dd, qq




class additive_term_rho1(torch.autograd.Function):
    """
    additive term rho1, rho1 = 1.0/(2*ad) : ad or add(,2) or d in mopac calpar.f
    """
    @staticmethod
    def forward(ctx, hsp_ev, D1):
        """
        hsp_ev : hsp in unit eV
        hsp_A = (s_A p_alpha_A, s_A p_alpha_A) = [mu_alpha_A, mu_alpha_A]
        hsp = [mu_pi, mu_pi]
        rho1 =  1/(2d)
        hsp = e^2 ( d/2 - 1/2/sqrt( 4 * D1^2 + 1/d^2))

        D1 : dipole charge separation, ( dd in mopac )
        D1: in atomic unit
        hsp in atomic units

        hsp_ev : shape (n_atoms,)
        D1 : shape (n_atoms,)

        return rho1 : additive term rho1
        """
        if hsp_ev.dtype==torch.float32:
            eps = 1.0e-7
        elif hsp_ev.dtype==torch.float64:
            eps = 1.0e-16
        hsp = hsp_ev/ev # change to atomic units

        #d1 = (hsp/D1**2)**(1.0/3.0) # lowest order approximation
        c=hsp<0.0
        d1 = (torch.abs(hsp)/D1**2)**(1.0/3.0)
        d1[c]*=-1.0

        d2 = d1+0.04
        for i in range(1,6):
            hsp1 = 0.5*d1 - 0.5/torch.sqrt(4.0*D1**2+1.0/d1**2)
            hsp2 = 0.5*d2 - 0.5/torch.sqrt(4.0*D1**2+1.0/d2**2)

            d3 = torch.where(torch.abs(hsp2-hsp1)>eps,d1 + (d2-d1)*(hsp-hsp1)/(hsp2-hsp1),d2)
            #d3 = d1 + (d2-d1)*(hsp-hsp1)/(hsp2-hsp1)
            d1 = d2
            d2 = d3
            #print(d1, d2)
        #ad = d2
        #rho1 = 0.5/d
        #print(d2)

        rho1 = 0.5/d2
        """
        if torch.isnan(rho1).any():
            c=torch.isnan(rho1)
            print("inside rho1")
            print(rho1[c])
            print(d2[c])
            print(hsp_ev[c])
            print(D1[c])
            print(hsp_ev[~c])
            print(D1[~c])
        """
        del hsp, d1, d2, d3, hsp1, hsp2
        ctx.save_for_backward(rho1, D1)

        return rho1

    @staticmethod
    def backward(ctx, grad_output):
        """
        hsp = e^2 ( d/2 - 1/2/sqrt( 4 * D1^2 + 1/d^2))
        hsp = e^2 * (1/4) * (1/rho1 - 1/sqrt( D1^2 + rho1^2 ))
        # Atomic unit
        dhsp = (1/4) * ( rho1 / (D1^2+rho1^2)^(3/2) - 1/rho1^2 ) * drho1
               + (1/4) * D1/(D1^2+rho1^2)^(3/2) * dD1
        dhsp/drho1 = (1/4) * ( rho1 / (D1^2+rho1^2)^(3/2) - 1/rho1^2 )
        dD1/drho1 = (D1^2+rho1^2)^(3/2)/rho1^2/D - rho1/D1
        """
        rho1, D1 = ctx.saved_tensors
        tmp = (D1**2 + rho1**2)**(1.5)
        # dhsp/drho1 * grad_rho1
        # change from atomic unit to eV for dhsp
        dhsp = 4.0/(rho1/tmp-1.0/rho1**2)*grad_output/ev
        dD1 = grad_output/(tmp/rho1**2/D1 - rho1/D1)
        del tmp
        return (dhsp, dD1)


class additive_term_rho2(torch.autograd.Function):
    """
    additive term rho2, rho2 = 1.0/(2*aq) : aq or add(,3) or q in mopac calpar.f
    """
    @staticmethod
    def forward(ctx, hpp_ev, D2):
        """
        hpp_ev : hpp in unit eV
        hpp_A = (p_alpha_A p_beta_A, p_alpha_A p_beta_A) = [Q_alpha__beta_A, Q_alpha__beta_A]
        hpp = [Q_xy, Q_xy]
        rho2 =  1/(2q)
        hpp = e^2 ( q/4 - 1/2/sqrt( 4 * D2^2 + 1/q^2) + 1/4/sqrt(8D2^2 + 1/q^2) )
        D2 : dipole charge separation, ( qq in mopac )
        D2: in atomic unit
        hspp in atomic units

        hpp_ev : shape (n_atoms,)
        D2 : shape (n_atoms,)

        return rho2 : additive term rho2
        """
        if hpp_ev.dtype==torch.float32:
            eps = 1.0e-7
        elif hpp_ev.dtype==torch.float64:
            eps = 1.0e-16
        hpp = hpp_ev/ev

        #q1 = (hpp/3.0/D2**4)**0.2
        c=hpp<0.0
        q1=(torch.abs(hpp)/3.0/D2**4)**0.2
        q1[c]*=-1.0

        q2 = q1 + 0.04
        for i in range(1,6):
            hpp1 = 0.25*q1 - 0.5/torch.sqrt(4.0*D2**2+1.0/q1**2) + 0.25/torch.sqrt(8.0*D2**2+1.0/q1**2)
            hpp2 = 0.25*q2 - 0.5/torch.sqrt(4.0*D2**2+1.0/q2**2) + 0.25/torch.sqrt(8.0*D2**2+1.0/q2**2)
            #q3 =  q1 + (q2-q1)*(hpp-hpp1)/(hpp2-hpp1)
            q3 = torch.where(torch.abs(hpp2-hpp1)>eps,q1 + (q2-q1)*(hpp-hpp1)/(hpp2-hpp1),q2)
            q1 = q2
            q2 = q3
        # aq = q3
        # rho2 = 0.5/q
        rho2 = 0.5/q2
        """
        if torch.isnan(rho2).any():
            c=torch.isnan(rho2)
            print("inside rho2")
            print(rho2[c])
            print(q2[c])
            print(hpp_ev[c])
            print(D2[c])
            print(hpp_ev[~c])
            print(D2[~c])
        """
        del hpp, q1, q2, q3, hpp1, hpp2
        ctx.save_for_backward(rho2, D2)
        return rho2

    @staticmethod
    def backward(ctx, grad_output):
        """
        hpp = e^2 ( q/4 - 1/2/sqrt( 4 * D2^2 + 1/q^2) + 1/4/sqrt(8D2^2 + 1/q^2) )
        # AU
        hpp = 1/8/rho2 - 1/4/sqrt(D2^2+rho2^2) + 1/8/sqrt(2*D2^2+rho2^2)
        dhpp =  -1/8/rho2^2 drho2 + 1/4/(D2^2+rho2^2)^(3/2) * (D2*dD2 + rho2*drho2)
               - 1/8/(2*D2^2+rho2^2)^(3/2) * (2*D2*DD2 + rho2 * drho2)
        dhpp/drho2 = -1/8/rho2^2 + rho2/4/(D2^2+rho2^2)^(3/2) -rho2/8/(2*D2^2+rho2^2)^(3/2)
        dD2/drho2 = - (dhpp/drho2)/(dhpp/dD2)
        dhpp/dD2 = D2/4/(D2^2+rho2^2)^(3/2) - D2/4/(2*D2^2+rho2^2)^(3/2)
        """

        # 1 hatree  = 27.21 eV
        rho2, D2 = ctx.saved_tensors
        tmp1 = 1.0/(D2**2 + rho2**2)**1.5
        tmp2 = 1.0/(2.0*D2**2 + rho2**2)**1.5
        dhppdrho2 = -0.125/rho2**2 + rho2*(tmp1/4.0-tmp2/8.0)
        dhpp_ev = grad_output/dhppdrho2/ev
        dD2 = -(D2/4.0*(tmp1-tmp2))*grad_output/dhppdrho2
        del tmp1, tmp2, dhppdrho2
        return (dhpp_ev, dD2)
