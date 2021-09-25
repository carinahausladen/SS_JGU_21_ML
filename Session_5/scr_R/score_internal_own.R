function (type = "valid", fuzzy = FALSE, ground.truth = NULL) 
{
  if (fuzzy) {
    external <- c("RI", "ARI", "VI", "NMIM")
    internal <- c("MPC", "K", "T", "SC", "PBMF")
    minimize <- c("VI", "K", "T")
  }
  else {
    external <- c("RI", "ARI", "J", "FM", "VI")
    internal <- c("Sil", "D", "COP", "DB", "DBstar", "CH", 
                  "SF")
    minimize <- c("VI", "COP", "DB", "DBstar")
  }
  if ("valid" %in% type) {
    type <- if (is.null(ground.truth)) 
      internal
    else c(external, internal)
  }
  else if ("external" %in% type) {
    if (is.null(ground.truth)) 
      stop("The ground.truth is needed for external CVIs.")
    type <- external
  }
  else if ("internal" %in% type) {
    type <- internal
  }
  else {
    type <- match.arg(type, c(external, internal), several.ok = TRUE)
    if (any(external %in% type) && is.null(ground.truth)) 
      stop("The ground.truth is needed for external CVIs.")
  }
  majority <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
  }
  score <- function(objs, ...) {
    do.call(rbind, lapply(objs, function(obj) {
      internal <- intersect(type, internal)
      external <- intersect(type, external)
      if (length(internal) > 0L) 
        cvis <- cvi(a = obj, type = internal, ...)
      else cvis <- numeric()
      if (length(external) > 0L) {
        if (fuzzy) 
          cvis <- c(cvis, cvi(obj@fcluster, ground.truth, 
                              external, ...))
        else cvis <- c(cvis, cvi(obj@cluster, ground.truth, 
                                 external, ...))
      }
  #    minimized <- names(cvis) %in% minimize
  #    if (length(minimized) > 0L && any(minimized)) 
  #      cvis[minimized] <- 1/cvis[minimized]
      cvis
    }))
  }
  pick <- function(results, objs, ...) {
    objs_missing <- missing(objs)
    best_by_type <- sapply(results, function(result) {
      score <- result[type]
      best_by_cvi <- apply(score, 2L, which.max)
      if (length(type) > 1L && length(unique(best_by_cvi)) == 
          length(best_by_cvi)) 
        stop("All votes are distinct, so majority voting is inconclusive.")
      majority(best_by_cvi)
    })
    best_overall <- Map(results, best_by_type, f = function(result, 
                                                            row_id) {
      result[row_id, type, drop = FALSE]
    })
    best_overall <- do.call(rbind, best_overall)
    best_overall <- apply(best_overall, 2L, which.max)
    if (length(type) > 1L && length(unique(best_overall)) == 
        length(best_overall)) 
      stop("All votes are distinct, so majority voting is inconclusive.")
    best_overall <- majority(best_overall)
    if (objs_missing) 
      results[[best_overall]][best_by_type[best_overall], 
                              , drop = FALSE]
    else list(object = objs[[best_overall]][[best_by_type[best_overall]]], 
              config = results[[best_overall]][best_by_type[best_overall], 
                                               , drop = FALSE])
  }
  list(score = score, pick = pick)
}