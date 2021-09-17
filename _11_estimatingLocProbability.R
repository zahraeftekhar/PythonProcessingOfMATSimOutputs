rm(list = ls())
library(rethinking)
library(gtools) 
library(rstan)
library(rjags)
library(data.table)
library(R2OpenBUGS)
library(coda)
library(lattice)
library(hesim)
savingLoc = "Y:/ZahraEftekhar/phase4/"
amsterdamMezuroZones = data.table(read.csv(paste0(savingLoc,"amsterdamMezuroZones.CSV"))[ ,c('mzr_id')])
colnames(amsterdamMezuroZones) <- c("location")
amsterdamMezuroZones$code <- 1:dim(amsterdamMezuroZones)[1]
amsterdamMezuroZones <- amsterdamMezuroZones[order(amsterdamMezuroZones$location),]
ACTS <-  c("Home","Work","Other")
acts <- c("home","work","other")
prop <- 0.1
data_all <- data.frame()
for (ACT in ACTS){
  data_act = assign(paste0("data_",ACT),data.table(read.csv(paste0(savingLoc,"training",ACT,'_prop',prop,'.CSV'))))
  data_all <- rbind(data_all,data_act)
}
data_all$actCategory <- 3
for (act in acts[1:length(acts)-1]){
  
  data_all$actCategory[data_all$activityType == act] <-
    data_all$activityType[data_all$activityType == act]
}

# 
# 
# data_home <- read.csv("D:/ax/gis/phase2/train_home_zoneLabled.csv"
# )[ ,c("VEHICLE","activityType"
#       ,"start","duration","x","y","mzr_id")]
# data_work <- read.csv("D:/ax/gis/phase2/train_work_zoneLabled.csv"
# )[ ,c("VEHICLE","activityType"
#       ,"start","duration","x","y","mzr_id")]
# data_other <- read.csv("D:/ax/gis/phase2/train_other_zoneLabled.csv"
# )[ ,c("VEHICLE","activityType"
#       ,"start","duration","x","y","mzr_id")]
# 
# 
# 
# data_all <- rbind(data_home,data_work,data_other)
# data_all$actCategory <- 3
# data_all$actCategory[data_all$activityType == "home"] <-
#   data_all$activityType[data_all$activityType == "home"]
# data_all$actCategory[data_all$activityType == "work"] <-
#   data_all$activityType[data_all$activityType == "work"]
# data_all$actCategory <- lapply(data_all$actCategory,as.integer)

# boxplot(actCategory ~ mzr_id, data=data_all)
lmod = lm(actCategory ~ mzr_id, data=data_all)
summary(lmod)
anova(lmod)
png(file="D:/ax/gis/phase2/activityDistribution.png",width=900, height=500)
boxplot(actCategory~mzr_id, data=data_all,col="grey90",xlab="OD zone"
        , ylab="activity category", xaxt="n", yaxt="n", cex.lab=1.6)
ticks<-c(1,1.5,2,2.5,3)
ll <- c("home"," ","work"," ","other")
axis(2,at=ticks,labels=ll,cex.axis = 1.4, line =0.6)
dev.off()

dattt <- data_all$mzr_id
dattt <- cbind(dattt,data_all$actCategory)
colnames(dattt) <- c("location","actCategory")
dattt <- data.table(dattt)
setkey(dattt,location)
codes <- as.data.frame(sort(unique(dattt$location)))
codes$code <- 1:length(unique(dattt$location))
colnames(codes) <- c("location","code")
codes <- data.table(codes)
setkey(codes,location)

dattt$location <- as.integer(dattt$location)
dat <- merge(dattt,codes,by=c("location"),all = TRUE)
dat <- dat[order(dat$location),]
modelParams <- matrix(0,nrow = length(amsterdamMezuroZones$location),ncol = 5 )
modelParams[,1] <- amsterdamMezuroZones$location
modelParams <- modelParams[order(modelParams[,1]),]


#modelParams[which(modelParams[,1]== amsterdamMezuroZones$location[1]),2:5] <-pm_params
conc <- 1
denom <- dim(data_all)[1]
alpha<-c((length(which(data_all$actCategory==1))/denom)*conc,
         (length(which(data_all$actCategory==2))/denom)*conc,
         (length(which(data_all$actCategory==3))/denom)*conc)

for (loc in 1:length(amsterdamMezuroZones$location)){
  y <- c(dat$actCategory[dat$location==amsterdamMezuroZones$location[loc]])
  if (length(y)>1){
    N=length(y)


    dataList = list(
      y = y ,
      N = N , 
      alpha=alpha
    )
    
    #------------------------------------------------------------------------------
    # THE MODEL.
    model = function(){
      #likelihood
      for ( i in 1:N ) {
        y[i] ~ dcat( theta[] )
      }
      #prior
      theta[1:3] ~ ddirch( alpha[] )
      # alpha[1:3] ~ dgamma(2, 0.2)
      
    }
    model.file <- file.path(tempdir(),"model.txt") 
    write.model(model, model.file)
    
    #------------------------------------------------------------------------------
    # INTIALIZE THE CHAINS.
    
    # Specific initialization is not necessary in this case, 
    # but here is a lazy version if wanted:
    # initsList = list( theta=0.5 , m=1 ) 
    
    #------------------------------------------------------------------------------
    
    params = c("theta","alpha") 
    inits<-function(){theta=c(1/3,1/3,1/3)}
    set.seed(50)
    out<-bugs(dataList,inits,params,model.file,n.chains = 3
              ,n.iter=10000,codaPkg = TRUE,n.burnin = 1000,DIC = TRUE)
    out.coda <- read.bugs(out) 
    
    # xyplot(out.coda)
    # densityplot(out.coda)
    # acfplot(out.coda)
    # gelman.diag(out.coda) # we should perform the Gelman-Rubin convergence
    # diagnostic with gelman.diag. The shrink factors should be below 1.05.
    
    
    #  it is prudent to create the Gelman-Rubin-Brooks plot for visual 
    # confirmation of the shrink factor convergence as well.
    # gelman.plot(out.coda)
    # 
    # out.summary <- summary(out.coda, q=c(0.025, 0.975)) 
    # out.summary$stat["theta[1]",] 
    # out.summary$stat["theta[2]",] 
    # out.summary$stat["theta[3]",] 
    # out.summary$q["theta[1]", ] 
    
    
    mod_sim <- out.coda
    mod_csim <- as.mcmc(do.call(rbind,mod_sim))
    #plot(mod_sim,ask=TRUE)
    #dic <- dic.samples(mod,n.iter = 1e3)
    
    
    ## Model checking
    pm_params <- colMeans(mod_csim)
    # we use this for prediction
    #pm_params
    modelParams[loc,2:5] <-pm_params
    }
  
}

home <- data.table(table(data_Home$mzr_id))
colnames(home) <- c("location","freq")
home$location <- as.integer(home$location)
home <- merge(home,amsterdamMezuroZones,by="location",all=TRUE)
home$freq[is.na(home$freq)] = 0.001

work <- data.table(table(data_Work$mzr_id))
colnames(work) <- c("location","freq")
work$location <- as.integer(work$location)
work <- merge(work,amsterdamMezuroZones,by="location",all=TRUE)
work$freq[is.na(work$freq)] = 0.001

other <- data.table(table(data_Other$mzr_id))
colnames(other) <- c("location","freq")
other$location <- as.integer(other$location)
other <- merge(other,amsterdamMezuroZones,by="location",all=TRUE)
other$freq[is.na(other$freq)] = 0.001

data_allAct <- merge(home,work,by=c("code","location"))
data_allAct <- merge(data_allAct,other,by=c("code","location"))
colnames(modelParams) <- c("mzr_id","deviance","home","work","other")
write.csv(modelParams,paste0(savingLoc,"locProbability.csv"), row.names = FALSE)
