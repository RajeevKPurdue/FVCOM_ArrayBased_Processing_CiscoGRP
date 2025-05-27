# Mark Rowe 1-13-2022
# modified from version by Qianqian Liu to plot Chl or prob
# modified by RK (Purdue FNR- Höök Lab) to scale to GRP (g/g/d) values with diverging colors

library(fields)
library(wesanderson)  # Load the wesanderson package for the color palette

getColorsVal <- function(values, brk = seq(-0.015, 0.050, length.out = 7)) {
  require(fields)
  require(wesanderson)
  
  # values is a numeric vector
  tox1 <- values  # Filter out NA values
  brkm <- max(brk, na.rm = TRUE)
  label <- "Probability"  # Label for the color scale
  
  # Use the diverging color palette "The Life Aquatic"
  nlevel <- length(brk) - 1
  colorbar <- wes_palette("Zissou1", nlevel, type = "continuous")
  
  # Cut values into specified breaks
  values <- cut(tox1, breaks = brk, include.lowest = TRUE)
  
  # Set zlim to match the range of the number of levels
  zlim <- c(1, nlevel)
  
  # Generate colors for the values based on the colorbar and zlim
  cols <- fields::color.scale(as.numeric(values), col = colorbar, zlim = zlim, eps = 0.0, transparent.color = "gray")
  
  return(list(cols = cols, zlim = zlim, label = label, brk = brk, colorbar = colorbar))
}

# Example usage with a range from -0.015 to 0.050
values <- seq(-0.015, 0.050, length.out = 100)
color_info <- getColorsVal(values)

# Inspect the output
print(color_info)

######### function to plot color scale
# http://menugget.blogspot.de/2013/12/new-version-of-imagescale-function.html
# copied from the website on 10-27-2015

#This function creates a color scale for use with the image()
#function. Input parameters should be consistent with those
#used in the corresponding image plot. The "axis.pos" argument
#defines the side of the axis. The "add.axis" argument defines
#whether the axis is added (default: TRUE)or not (FALSE).
image.scale <- function(z, zlim, col = heat.colors(12), las=1, brkLabs=TRUE, at=NULL,
                        breaks, axis.pos=1, add.axis=TRUE, ...){
  if(!missing(breaks)){
    if(length(breaks) != (length(col)+1)){stop("must have one more break than colour")}
  }
  #   if(!missing(brkLabs)){
  #     brkLabs <- TRUE
  #   }
  if(missing(breaks) & !missing(zlim)){
    breaks <- seq(zlim[1], zlim[2], length.out=(length(col)+1)) 
  }
  if(missing(breaks) & missing(zlim)){
    zlim <- range(z, na.rm=TRUE)
    zlim[2] <- zlim[2]+c(zlim[2]-zlim[1])*(1E-3)#adds a bit to the range in both directions
    zlim[1] <- zlim[1]-c(zlim[2]-zlim[1])*(1E-3)
    breaks <- seq(zlim[1], zlim[2], length.out=(length(col)+1))
  }
  poly <- vector(mode="list", length(col))
  for(i in seq(poly)){
    poly[[i]] <- c(breaks[i], breaks[i+1], breaks[i+1], breaks[i])
  }
  if(axis.pos %in% c(1,3)){ylim<-c(0,1); xlim<-range(breaks)}
  if(axis.pos %in% c(2,4)){ylim<-range(breaks); xlim<-c(0,1)}
  plot(1,1,t="n",ylim=ylim, xlim=xlim, axes=FALSE, xlab="", ylab="", xaxs="i", yaxs="i", ...)  
  for(i in seq(poly)){
    if(axis.pos %in% c(1,3)){
      polygon(poly[[i]], c(0,0,1,1), col=col[i], border=col[i])
    }
    if(axis.pos %in% c(2,4)){
      polygon(c(0,0,1,1), poly[[i]], col=col[i], border=col[i])
    }
  }
  box()
  if(add.axis) {axis(axis.pos, las=las, labels=brkLabs, at=at)}
}

