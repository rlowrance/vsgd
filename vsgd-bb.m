% vsgd-bb.m
% from Sixin Zhang 2012-08-12
% INIT lg
lg = zeros(length(theta),1);
lg2 = zeros(L,1);
init_nb = floor(n*options.minimizer_pd);
perms = randperm(n);
for id = 1 : init_nb
    [~,grad] = model(theta, data(:,perms(id)), options);
    lg = lg + grad;
    for i = 1 : L
        lg2(i) = lg2(i) + sum(grad(Ibegin(i):Iend(i)).^2);
    end
end
lg = lg ./ init_nb;
lg2 = lg2 ./ init_nb;
lg2 = lg2 .* options.minimizer_bcv;

lrv = zeros(length(theta),1);
for i = 1 : L
    lrv(Ibegin(i):Iend(i)) = sum(lg(Ibegin(i):Iend(i)).^2)./lg2(i);
end

% INIT alpha
alpha = zeros(length(theta),1);

% INIT lh
[~,~,lh] = model(theta,data(:,perms(1:init_nb)),options);
lh = lh .* options.minimizer_bcdh;

lrh = zeros(length(theta),1);
for i = 1 : L
    lrh(Ibegin(i):Iend(i)) = max(lh(Ibegin(i):Iend(i)));
end

% TRAIN by mini batch (should loop)
mini = data(:,bs*perms(k)-(bs-1):bs*perms(k));
[loss,grad,dhess] = model(theta,mini,options);

alpha = (1-lrv)./(2-lrv-alpha);
alpha = max(sqrt(eps),alpha);
alpha = min(1-sqrt(eps),alpha);

lg = alpha.*lg + (1-alpha).*grad;
for i = 1 : L
    lg2(i) = ...
        alpha(Ibegin(i)).*lg2(i) + ...
        (1-alpha(Ibegin(i))).*sum(grad(Ibegin(i):Iend(i)).^2);
    lrv(Ibegin(i):Iend(i)) = sum(lg(Ibegin(i):Iend(i)).^2)./lg2(i);
end

lh = alpha.*lh + (1-alpha).*dhess;
lrh = zeros(length(theta),1);
for i = 1 : L
    lrh(Ibegin(i):Iend(i)) = max(lh(Ibegin(i):Iend(i)));
end

lr = lrv ./ lrh;
theta = theta - lr.*grad;
