#----------------------------------------------------------------------
#                                                                      
#                           CERN                                       
#                                                                      
#     European Organization for Nuclear Research                       
#                                                                      
#     
#     This file is part of the code:
#                                                                       
#                   PyPIC Version 2.4.5                   
#                  
#                                                                       
#     Author and contact:   Giovanni IADAROLA 
#                           BE-ABP Group                               
#                           CERN                                       
#                           CH-1211 GENEVA 23                          
#                           SWITZERLAND  
#                           giovanni.iadarola@cern.ch                  
#                                                                      
#                contact:   Giovanni RUMOLO                            
#                           BE-ABP Group                               
#                           CERN                                      
#                           CH-1211 GENEVA 23                          
#                           SWITZERLAND  
#                           giovanni.rumolo@cern.ch                    
#                                                                      
#
#                                                                      
#     Copyright  CERN,  Geneva  2011  -  Copyright  and  any   other   
#     appropriate  legal  protection  of  this  computer program and   
#     associated documentation reserved  in  all  countries  of  the   
#     world.                                                           
#                                                                      
#     Organizations collaborating with CERN may receive this program   
#     and documentation freely and without charge.                     
#                                                                      
#     CERN undertakes no obligation  for  the  maintenance  of  this   
#     program,  nor responsibility for its correctness,  and accepts   
#     no liability whatsoever resulting from its use.                  
#                                                                      
#     Program  and documentation are provided solely for the use  of   
#     the organization to which they are distributed.                  
#                                                                      
#     This program  may  not  be  copied  or  otherwise  distributed   
#     without  permission. This message must be retained on this and   
#     any other authorized copies.                                     
#                                                                      
#     The material cannot be sold. CERN should be  given  credit  in   
#     all references.                                                  
#----------------------------------------------------------------------



from numpy import squeeze, array,diff, max, sum, sqrt,\
                  logical_and, logical_or, ones, zeros, take, arctan2, sin, cos, isnan
import numpy as np
import scipy.io as sio

def generate_rectangular_chamber(x_grid, y_grid):
    xmin = np.min(x_grid)
    xmax = np.max(x_grid)
    ymin = np.min(y_grid)
    ymax = np.max(y_grid)

    # Rectangle vertices (counter-clockwise)
    Vx = np.array([xmin, xmax, xmax, xmin])
    Vy = np.array([ymin, ymin, ymax, ymax])

    # Semi-axes of the inscribed ellipse
    cx = 0.5 * (xmax - xmin)
    cy = 0.5 * (ymax - ymin)

    chamber_dict = {
        "Vx": Vx,
        "Vy": Vy,
        "x_sem_ellip_insc": cx,
        "y_sem_ellip_insc": cy,
    }

    return polyg_cham_geom_object(chamber_dict)

class polyg_cham_geom_object(object):
    def __init__(self, filename_chm, flag_non_unif_sey=False, 
                 flag_verbose_file=False, flag_verbose_stdout=False):
        
        if type(filename_chm)==str:
            dict_chm=sio.loadmat(filename_chm)
        else:
            dict_chm=filename_chm

        Vx=squeeze(dict_chm['Vx'])
        Vy=squeeze(dict_chm['Vy'])
        cx=float(squeeze(dict_chm['x_sem_ellip_insc']))
        cy=float(squeeze(dict_chm['y_sem_ellip_insc']))
        
        if flag_non_unif_sey==1:
            self.del_max_segments = squeeze(dict_chm['del_max_segments'])
            self.R0_segments = squeeze(dict_chm['R0_segments'])
            self.Emax_segments = squeeze(dict_chm['Emax_segments'])
        
        self.N_vert=len(Vx)
        
        N_edg=len(Vx)

        Vx=list(Vx)
        Vy=list(Vy)
        
        Vx.append(Vx[0])
        Vy.append(Vy[0])
        
        Vx=array(Vx)
        Vy=array(Vy)
        
        Nx=-diff(Vy,1)
        Ny=diff(Vx,1)
        
        norm_N=sqrt(Nx**2+Ny**2) 
        Nx=Nx/norm_N
        Ny=Ny/norm_N
        
        self.x_aper = max(abs(Vx))
        self.y_aper = max(abs(Vy))
        self.Vx=Vx
        self.Vy=Vy
        self.Nx=Nx
        self.Ny=Ny
        self.N_edg=N_edg
        self.cx=cx
        self.cy=cy
        
        self.N_mp_impact=0
        self.N_mp_corrected=0
        self.chamb_type='polyg'
        
        self.flag_verbose_stdout = flag_verbose_stdout
        self.flag_verbose_file = flag_verbose_file
        
        if self.flag_verbose_file:
            fbckt=open('bcktr_errors.txt','w')
            fbckt.write('kind,x_in,y_in,x_out, y_out\n')
            fbckt.close()

#    def is_outside(self, x_mp, y_mp):
#        N_pts=len(x_mp)
#        flag_inside=array(N_pts*[True])
#        for ii in xrange(self.N_edg):
#            flag_inside[flag_inside]=((y_mp[flag_inside]-self.Vy[ii])*(self.Vx[ii+1]-self.Vx[ii])\
#                                      -(x_mp[flag_inside]-self.Vx[ii])*(self.Vy[ii+1]-self.Vy[ii]))>0
#
#        return ~flag_inside
    
    def is_outside(self, x_mp, y_mp):
        
        flag_outside=(((x_mp/self.cx)**2 + (y_mp/self.cy)**2)>1)
        #flag_outside=array([True]*len(x_mp))
        
        if flag_outside.any():
            x_mp_chk=x_mp[flag_outside]  
            y_mp_chk=y_mp[flag_outside]
            N_pts=len(x_mp_chk)
            flag_inside_chk=array(N_pts*[True])
            for ii in range(self.N_edg):
                flag_inside_chk[flag_inside_chk]=((y_mp_chk[flag_inside_chk]-self.Vy[ii])*(self.Vx[ii+1]-self.Vx[ii])\
                                          -(x_mp_chk[flag_inside_chk]-self.Vx[ii])*(self.Vy[ii+1]-self.Vy[ii]))>0
            flag_outside[flag_outside]=~flag_inside_chk
            
        return flag_outside
    
    def impact_point_and_normal(self, x_in, y_in, z_in, x_out, y_out, z_out, resc_fac=0.99, flag_robust=True):
        
        N_impacts=len(x_in)
        t_min = ones(N_impacts)+1.
        i_found=array(N_impacts*[-1])
        mask_found=array(N_impacts*[False])
        
        self.N_mp_impact=self.N_mp_impact+N_impacts
        
        
        for ii in range(self.N_edg):
            t_curr = (self.Nx[ii]*(self.Vx[ii]-x_in)+self.Ny[ii]*(self.Vy[ii]-y_in)) / \
                             (self.Nx[ii]*(x_out-x_in)+self.Ny[ii]*(y_out-y_in))
            
            mask_updatemin =  logical_and(t_curr>=0, t_curr<t_min)
            t_min[mask_updatemin]=t_curr[mask_updatemin]
            mask_found=logical_or(mask_updatemin,mask_found) 
            i_found[mask_updatemin]=ii
            
            mask_same_min=(t_curr==t_min)
            if mask_same_min.any():
                mask_upd_i_found=array(N_impacts*[False])
                t_border=((y_out[mask_same_min]-y_in[mask_same_min])*(x_in[mask_same_min]-self.Vx[ii])+(x_in[mask_same_min]-x_out[mask_same_min])*(y_in[mask_same_min]-self.Vy[ii])) / \
                         ((y_out[mask_same_min]-y_in[mask_same_min])*(self.Vx[ii+1]-self.Vx[ii])+(x_in[mask_same_min]-x_out[mask_same_min])*(self.Vy[ii+1]-self.Vy[ii]))
                mask_upd_i_found[mask_same_min] = logical_and(t_border>=0., t_border<=1.)
                i_found[mask_upd_i_found]=ii
            
            
        t_min=resc_fac*t_min;
        x_int=t_min*x_out+(1.-t_min)*x_in;
        y_int=t_min*y_out+(1.-t_min)*y_in;
        z_int=0*t_min
        
        Nx_int = zeros(N_impacts)
        Ny_int = zeros(N_impacts)
        Nx_int[mask_found]=take(self.Nx, i_found[mask_found])
        Ny_int[mask_found]=take(self.Ny, i_found[mask_found])
        
        if sum(mask_found)<N_impacts:
            mask_not_found = ~mask_found
            
            x_int[mask_not_found] = x_in[mask_not_found];
            y_int[mask_not_found] = y_in[mask_not_found];
            
            #compute some kind of normal ....
            par_cross=arctan2(self.cx*y_in[mask_not_found],self.cy*x_int[mask_not_found]);

            Dx=-self.cx*sin(par_cross);
            Dy=self.cy*cos(par_cross);
            
            Nx_corr=-Dy;
            Ny_corr=Dx;
            
            neg_flag=((Nx_corr*x_int[mask_not_found]+Ny_corr*y_int[mask_not_found])>0);
            
            Nx_corr[neg_flag]=-Nx_corr[neg_flag];
            Ny_corr[neg_flag]=-Ny_corr[neg_flag];
            
            Nx_int[mask_not_found]=Nx_corr
            Ny_int[mask_not_found]=Ny_corr
            

            x_in_error = x_in[mask_not_found]
            y_in_error = y_in[mask_not_found]
            x_out_error = x_out[mask_not_found]
            y_out_error = y_out[mask_not_found]
            N_errors = len(x_in_error)
            self.N_mp_corrected = self.N_mp_corrected + N_errors
            
            if self.flag_verbose_stdout:
                print("""Reporting backtrack error of kind 1: no impact found""")
                print("""x_in, y_in, x_out, y_out""")
                for i_err in range(N_errors):
                    lcurr = '%.10e,%.10e,%.10e,%.10e'%(x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                    print(lcurr)
                print("""End reporting backtrack error of kind 1""")
  
            if self.flag_verbose_file:
                with open('bcktr_errors.txt','a') as fbckt:
                    for i_err in range(N_errors):
                        lcurr = '%.10e,%.10e,%.10e,%.10e'%(x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                        fbckt.write('1,'+lcurr+'\n')
  

            
            
        if flag_robust:
            flag_impact=self.is_outside(x_int, y_int)
            if flag_impact.any(): 
                self.N_mp_corrected = self.N_mp_corrected + sum(flag_impact)
                x_int[flag_impact] = x_in[flag_impact];
                y_int[flag_impact] = y_in[flag_impact];
                x_in_error = x_in[flag_impact]
                y_in_error = y_in[flag_impact]
                x_out_error = x_out[flag_impact]
                y_out_error = y_out[flag_impact]
                N_errors = len(x_in_error)
                    
                if self.flag_verbose_stdout:
                    print("""Reporting backtrack error of kind 2: outside after backtracking""")
                    print("""x_in, y_in, x_out, y_out""")
                    for i_err in range(N_errors):
                        lcurr = '%.10e,%.10e,%.10e,%.10e'%(x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                        print(lcurr)
                    print("""End reporting backtrack error of kind 2""")
      
                if self.flag_verbose_file:
                    with open('bcktr_errors.txt','a') as fbckt:
                        for i_err in range(N_errors):
                            lcurr = '%.10e,%.10e,%.10e,%.10e'%(x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                            fbckt.write('2,'+lcurr+'\n')
             
            flag_impact=self.is_outside(x_int, y_int)
            if sum(flag_impact)>0:   
                #~ import pylab as pl
                #~ pl.close('all')
                #~ pl.plot(self.Vx, self.Vy)
                #~ pl.plot(x_in, y_in,'.b')
                #~ pl.plot(x_out, y_out,'.k')
                #~ pl.plot(x_int, y_int,'.g')
                #~ pl.plot(x_int[flag_impact], y_int[flag_impact],'.r')
                #~ pl.show()
                if self.flag_verbose_stdout:     
                    print("""Reporting backtrack error of kind 3: outside after correction""")
                    print("""x_in, y_in, x_out, y_out""")
                x_in_error = x_in[flag_impact]
                y_in_error = y_in[flag_impact]
                x_out_error = x_out[flag_impact]
                y_out_error = y_out[flag_impact]
                N_errors = len(x_in_error)

                if self.flag_verbose_stdout:
                    print("""Reporting backtrack error of kind 3: outside after correction""")
                    print("""x_in, y_in, x_out, y_out""")
                    for i_err in range(N_errors):
                        lcurr = '%.10e,%.10e,%.10e,%.10e'%(x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                        print(lcurr)
                    print("""End reporting backtrack error of kind 3""")
      
                if self.flag_verbose_file:
                    with open('bcktr_errors.txt','a') as fbckt:
                        for i_err in range(N_errors):
                            lcurr = '%.10e,%.10e,%.10e,%.10e'%(x_in_error[i_err], y_in_error[i_err], x_out_error[i_err], y_out_error[i_err])
                            fbckt.write('3,'+lcurr+'\n')   
                    
                
                raise ValueError('Outside after backtracking!!!!')
            
        
        return  x_int,y_int,z_int,Nx_int,Ny_int, i_found
        

class ellip_cham_geom_object:
    def __init__(self, x_aper, y_aper, flag_verbose_file=True):
        self.x_aper = x_aper
        self.y_aper = y_aper
        
        self.N_mp_impact=0
        self.N_mp_corrected=0
        self.chamb_type='ellip'
        
        self.flag_verbose_file = flag_verbose_file

    
    def is_outside(self, x_mp, y_mp):
        return (((x_mp/self.x_aper)**2 + (y_mp/self.y_aper)**2)>=1);
    
    def impact_point_and_normal(self, x_in, y_in, z_in, x_out, y_out, z_out, resc_fac=0.99, flag_robust=True):
    
        self.N_mp_impact=self.N_mp_impact+len(x_in)
        
        a=self.x_aper
        b=self.y_aper
        
        x_insq=x_in*x_in;
        y_insq=y_in*y_in;
        x_outsq=x_out*x_out;
        y_outsq=y_out*y_out;
        y_in__y_out=y_in*y_out;
        x_in__x_out=x_in*x_out;
        
        
        t0=(a**2*y_insq - a**2*y_in__y_out + \
          sqrt(a**4*b**2*y_insq - 2*a**4*b**2*y_in__y_out \
              + a**4*b**2*y_outsq +  \
            a**2*b**4*x_insq - 2*a**2*b**4*x_in__x_out +  \
            a**2*b**4*x_outsq - a**2*b**2*x_insq* \
            y_outsq + 2*a**2*b**2*x_in__x_out*y_in__y_out -  \
            a**2*b**2*x_outsq*y_insq) + b**2*x_insq -  \
            b**2*x_in__x_out)/(a**2*y_insq -  \
            2*a**2*y_in__y_out + a**2*y_outsq + b**2*x_insq -  \
            2*b**2*x_in__x_out + b**2*x_outsq);
        
        
    
        # Handle pathological cases
        mask_nan = isnan(t0)
        if sum(mask_nan)>0:
            t0[mask_nan]=0.
            if self.flag_verbose_file:
                x_nan=x_in[mask_nan]
                y_nan=y_in[mask_nan]
                fbckt=open('bcktr_errors.txt','a')
                for ii_bk in range(len(y_nan)):
                    fbckt.write('%e\t%e\tnan\n'%(x_nan[ii_bk],y_nan[ii_bk]))
                fbckt.close()
        
        t0=resc_fac*t0;
           
        t0[t0<1.e-2]=0;
        
        flag_ident=(((x_in-x_out)*(x_in-x_out)+(y_in-y_out)*(y_in-y_out))/(x_out*x_out+y_out*y_out))<1e-8;
        t0[flag_ident]=0;
        
       
        
        if sum(abs(t0.imag))>0:
            print('imag detected')
            raise ValueError('Backtracking: complex t0!!!!')
        
        x_int=t0*x_out+(1-t0)*x_in;
        y_int=t0*y_out+(1-t0)*y_in;
        z_int=t0*z_out+(1-t0)*z_in;
        
        
        if flag_robust:
            flag_impact=(((x_int/a)**2 + (y_int/b)**2)>1);
            
            
            if flag_impact.any():
                self.N_mp_corrected = self.N_mp_corrected + sum(flag_impact)
                ntrials=10
                while (sum(flag_impact)>0 and ntrials>0):
                    t0[flag_impact]=0.9*t0[flag_impact];
                    x_int=t0*x_out+(1-t0)*x_in;
                    y_int=t0*y_out+(1-t0)*y_in;
                    z_int=t0*z_out+(1-t0)*z_in;
                    
                    flag_impact=(((x_int/a)**2 + (y_int/b)**2)>1);
                    ntrials=ntrials-1
                
            flag_impact=(((x_int/a)**2 + (y_int/b)**2)>=1);
            if sum(flag_impact)>0:
                
                x_int_pat = x_int[flag_impact]
                y_int_pat = y_int[flag_impact]
                
                if self.flag_verbose_file:
                    fbckt=open('bcktr_errors.txt','a')
                    for ii_bk in range(len(x_int_pat)):
                        fbckt.write('%e\t%e\n'%(x_int_pat[ii_bk],y_int_pat[ii_bk]))
                
                    fbckt.close()
            
                
                x_pr=x_int_pat/a
                y_pr=y_int_pat/b
                
                r_pr=sqrt(x_pr**2+y_pr**2)
                
                x_pr=0.99*x_pr/r_pr
                y_pr=0.99*y_pr/r_pr
                
                x_int[flag_impact] = x_pr*a
                y_int[flag_impact] = y_pr*b
                
                
                
                
            flag_impact=(((x_int/a)**2 + (y_int/b)**2)>=1);
            if sum(flag_impact)>0:   
                print('err inside')
                raise ValueError('Outside after backtracking!!!!')
            
            
    
        
        
        par_cross=arctan2(a*y_int,b*x_int);
        
        Dx=-a*sin(par_cross);
        Dy=b*cos(par_cross);
        
        Nx=-Dy;
        Ny=Dx;
        
        neg_flag=((Nx*x_int+Ny*y_int)>0);
        
        Nx[neg_flag]=-Nx[neg_flag];
        Ny[neg_flag]=-Ny[neg_flag];
        
        
        nor=sqrt(Nx*Nx+Ny*Ny);
        
        Nx=Nx/nor;
        Ny=Ny/nor;
        
        i_found=None
        
        return  x_int,y_int,z_int,Nx,Ny, i_found
    
    def points_on_boundary(self, N_points):
        tt = np.linspace(0., 2*np.pi, N_points)
        return self.x_aper*cos(tt), self.y_aper*sin(tt)
