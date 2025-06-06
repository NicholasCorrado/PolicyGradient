cd ../.. # cd just outside the repo
tar --exclude="chtc" --exclude="configs" --exclude='grad_experiments' --exclude="plotting" --exclude='results' --exclude='.git' --exclude='.idea'  -czvf PROPS.tar.gz PROPS
scp PolicyGradient.tar.gz ncorrado@ap2001.chtc.wisc.edu:/staging/ncorrado
rm PolicyGradient.tar.gz