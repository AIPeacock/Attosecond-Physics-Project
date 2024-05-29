clear all;

dn=0.001;
nmax=200;

param=load('paramet.prn');
dt=param(5,1);
b=param(2,1);
omega_las=param(11,1);

E1=load('E.prn');
a=length(E1);
%E1(max(find(E1))+1:end) = [];
%b=length(E1);
%E(:,1)=E1;
E(1:b,1)=E1(1:b);

N_T=round(b*dt*omega_las/2/pi);
d_omega = 2*pi/(dt*b);

w1 = 0:d_omega:d_omega*(round(b/2)-1);
w2 = -d_omega*(round(b/2)):d_omega:-d_omega;
w11 = -w1';
w22 = -w2';
w(1:round(b/2),1)=w11;
w(round(b/2)+1:b,1)=w22;

E0_w=fft(E);
E_w(:,1)=E0_w;
sp_field=(abs(E0_w)).^2;

ww=-w11;
q=ww/omega_las;

tt=1:1:b;
time=tt'*dt;
Tt=time/2/pi*omega_las;

%mask_t(:,1)=exp(-((time(:,1)-75*2*pi/omega_las)/(1/2*dt*b*0.87)).^18);

sm=0.0015;
wm=15;
mask(1:length(ww),1)=1./(1+exp(-(ww(:,1)-d_omega*wm)/sm));
mask_=1./(1+exp((ww(:,1)-(ww(length(ww))-d_omega*(wm-1)))/sm));
mask_s(1:length(ww),1)=mask(1:length(ww),1);
mask_s(length(ww)+1:length(w),1)=mask_(1:length(ww),1);

%plot(mask_s);

Emasked=zeros(b,nmax);
d_t=zeros(b,nmax);
d_w=zeros(b,nmax);

for n=1:1:nmax
   
    n
    
   %FORTRAN gives d(t)
  
    system('Ar1.exe');
  
  %  pause(1);
     
  % loading dipole moment after the previous step
    
    dtmp=load('d.prn');
    dtmp(end+(b-length(dtmp)))=0;
    d_t(1:b,n)=dtmp(1:b)+E(1:b,n);
    
    
    d_w(:,n)=fft(d_t(:,n));
    
    % deleting from the dipole spectrum lowest frequences (and its reflections)  
    
    %tozero=16;
    %d_w(1:tozero,n)=0*d_w(1:tozero,n);
    %d_w(b-tozero+1:b,n)=0*d_w(b-tozero+1:b,n);
    
    d_w(:,n)=d_w(:,n).*mask_s(:,1);
          
   % d_w(1:round(N_T/2),n)=0*d_w(1:round(N_T/2),n);
   % d_w(b-round(N_T/2)+1:b,n)=0*d_w(b-round(N_T/2)+1:b,n);
      
  % integrating field spectrum
   
     E_w(1,n+1) = E_w(1,n);
     E_w(2:b,n+1) = E_w(2:b,n) - 1i*dn*d_w(2:b,n)./w(2:b,1); 
     E(:,n+1) = real(ifft(E_w(:,n+1)));
    
   % test with linear alpha 
 
 %   E_www(:,1)=E_w(:,1);
 %   E_www(:,n+1)=E_www(:,n) + 1i*(1*0+1i)*dn*E_www(:,n);
      
    %Emasked(:,n)=E(:,n+1).*mask(:,1);
    %Er=Emasked(:,n);
    Er=E(:,n+1);
    Er(end+(a-b))=0; 
    
    % writing new field in file to use it for next step
 
    fid1=fopen('E.prn','w');
    fprintf(fid1,'%2.3e\n', Er);
    fclose(fid1);
    
    %pause(2);
    
  %  sp_E(:,n+1)=(abs(E_w(:,n+1))).^2;
   % phase_E(:,n+1)=atan(imag(E_w(:,n+1))./real(E_w(:,n+1)));
     
end

s_f0=((abs(E_w)).^2);  %old_200

sp_n=s_f0(:,1:25:201);

semilogy(s_f0(:,2:2:11))

semilogy(s_f0(:,2))
hold on
semilogy(s_f0(:,51),'g-')
hold on
semilogy(s_f0(:,101),'m-')
hold on
semilogy(s_f0(:,201),'r-')
hold off

plot(s_f0(:,1))
hold on
plot(s_f0(:,51),'g-')
hold on
plot(s_f0(:,101),'g-')
hold on
plot(s_f0(:,201),'m-')
hold on
plot(s_f0(:,251),'r-')
hold off

s_0(1,:)=sum(s_f0(1901:1909,:));
s_0(2,:)=sum(s_f0(1947:1955,:));
s_0(3,:)=sum(s_f0(19:27,:));
s_0(4,:)=sum(s_f0(1:49,:));


nmax=200;
n=1:1:(nmax+1);
n=n';
loglog(n(2:(nmax+1),1),s_0(1,2:(nmax+1)),'r-',n(2:(nmax+1),1),s_0(2,2:(nmax+1)),'g-',n(2:(nmax+1),1),s_0(3,2:(nmax+1)),'b-')

loglog(n(2:(nmax+1),1),s_0(4,2:(nmax+1)),'r-')


plot(n(2:(nmax+1),1),s_0(1,2:(nmax+1)),'r-',n(2:(nmax+1),1),s_0(2,2:(nmax+1)),'g-',n(2:(nmax+1),1),s_0(3,2:(nmax+1)),'b-')


E_low_f=E_w;
E_low_f(49:length(E_w)-48,:)=0.*E_w(49:length(E_w)-48,:);
%E_low_f(45:68,:)=0.*E_w(45:68,:);
for k=1:1:nmax+1
E_low_t(:,k)=real(ifft(E_low_f(:,k)));
end

E_low=E_low_t(:,1:50:251);
semilogy(s_f1(:,1:6:31))

%semilogy(s_f1(:,2:5:37))

semilogy(s_f1(:,300))
hold on
semilogy(s_f1(:,11),'g-')
hold on
semilogy(s_f1(:,28),'m-')
hold on
semilogy(s_f1(:,201),'r-')
hold off

plot(q(:),s_f1(1:length(q),300),'k-')

semilogy(q(:),s_f1(1:length(q),300),'k-')
hold on
semilogy(q(:),s_f1(1:length(q),51))
hold on
semilogy(q(:),s_f1(1:length(q),101),'g-')
hold on
semilogy(q(:),s_f1(1:length(q),151),'y-')
hold on
semilogy(q(:),s_f1(1:length(q),201),'r-')
hold off

plot(Tt,E(:,2),'k-')
hold on
plot(Tt,E(:,51))
hold on
plot(Tt,E(:,101),'g-')
hold on
plot(Tt,E(:,151),'m-')
hold on
plot(Tt,E(:,201),'r-')
hold off


plot(E(round(b/N_T)*6+1:length(E0_w),2),'k-')
hold on
plot(E(round(b/N_T)*6+1:length(E0_w),21))
hold on
plot(E(round(b/N_T)*6+1:length(E0_w),31),'g-')
hold on
plot(E(round(b/N_T)*6+1:length(E0_w),37),'y-')
hold on
plot(E(round(b/N_T)*6+1:length(E0_w),51),'r-')
hold off


E_low_f=E_w;
E_low_f(50:length(E_w),:)=0.*E_w(50:length(E_w),:);
for k=1:1:nmax+1;
E_low_t(:,k)=real(ifft(E_low_f(:,k)));
end

%plot(Tt,E_low_t(:,1:4:21))

plot(Tt,E_low_t(:,2),'k-')
hold on
plot(Tt,E_low_t(:,11))
hold on
plot(Tt,E_low_t(:,21),'g-')
hold on
plot(Tt,E_low_t(:,26),'y-')
hold on
plot(Tt,E_low_t(:,31),'r-')
hold off


Er1=E(:,28);
Er1(end+(a-b))=0;

fid2=fopen('E_new.prn','w');
fprintf(fid2,'%2.3e\n', Er1);
fclose(fid2);

E_new=load('E_new.prn');
E_new1(1:b,1)=E_new(1:b);
Ew_new=fft(E_new1);

Ew_new2(:,1)=Ew_new(:,1);
Ew_new2(862:891,1)=(607.3/1.48/10)^0.5*Ew_new(862:891,1);
E11(:,1) = real(ifft(Ew_new2(:,1)));

Er2=E11(:,1);
Er2(end+(a-b))=0;

fid2=fopen('E_x0.1_masked.prn','w');
fprintf(fid2,'%2.3e\n', Er2);
fclose(fid2);

 plot(angle(E_w(1925,:)))


for kk=1:1:nmax
    energy(kk,1)=sum(s_f1(:,kk));
end    

plot(energy)


