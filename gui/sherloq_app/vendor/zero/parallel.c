/* SHERLOQ deterministic parallel adaptation of ZERO, AGPL-3.0-or-later.
 * Original authors and license: zero.c and LICENSE. */
#include <stdlib.h>
#include <math.h>
#include <dispatch/dispatch.h>
#include "zero.h"
#define TRUE 1
#define FALSE 0
struct Context { double *image; int X,Y; double cos_t[8][8]; unsigned char *zeros,*constant; int *votes; };
static void block_column(void *opaque,size_t column) {
 struct Context *ctx=opaque;int X=ctx->X,Y=ctx->Y,x=(int)column;
 for(int y=0;y<Y-7;y++) {
            int z = 0; /* number of zeros */
            int const_along_x = TRUE;
            int const_along_y = TRUE;

            /* check whether the block is constant along x or y axis */
            for (int xx=0; xx<8 && (const_along_x || const_along_y); xx++)
                for (int yy=0; yy<8 && (const_along_x || const_along_y); yy++) {
                    if (ctx->image[x+xx + (y+yy) * X] != ctx->image[x+0 + (y+yy) * X])
                        const_along_x = FALSE;
                    if (ctx->image[x+xx + (y+yy) * X] != ctx->image[x+xx + (y+0) * X])
                        const_along_y = FALSE;
                }

            /* compute DCT for 8x8 blocks staring at x,y and count its zeros */
            for (int i=0; i<8; i++)
                for (int j=0; j<8; j++)
                    if (i > 0 || j > 0) { /* coefficient 0 is not be counted */
                        double dct_ij = 0.0;

                        for (int xx=0; xx<8; xx++)
                            for (int yy=0; yy<8; yy++)
                                dct_ij += ctx->image[ x+xx + (y+yy) * X ]
                                    * ctx->cos_t[xx][i] * ctx->cos_t[yy][j];
                        dct_ij *= 0.25 * (i==0 ? 1.0/sqrt(2.0) : 1.0)
                            * (j==0 ? 1.0/sqrt(2.0) : 1.0);

                        /* the finest quantization in JPEG is to integer values.
                           in such case, the optimal threshold to decide if a
                           coefficient is zero or not is the midpoint between
                           0 and 1, thus 0.5 */
                        if (fabs(dct_ij) < 0.5) {
                            z++;
                        }
                    }


 ctx->zeros[x+y*X]=z;ctx->constant[x+y*X]=const_along_x||const_along_y;
 }
}
static void vote_row(void *opaque,size_t row) {
 struct Context *ctx=opaque;int X=ctx->X,Y=ctx->Y,y=(int)row;
 for(int x=0;x<X;x++) {
  int winner=-1,maximum=0,ties=0;
  if(x>=7 && y>=7 && x<X-7 && y<Y-7) {
   for(int xx=x-7;xx<=x;xx++)for(int yy=y-7;yy<=y;yy++) {
    int at=xx+yy*X,z=ctx->zeros[at];
    if(z==maximum)ties++;
    if(z>maximum){maximum=z;ties=1;winner=ctx->constant[at]?-1:(xx%8)+(yy%8)*8;}
   }
  }
  ctx->votes[x+y*X]=ties==1?winner:-1;
 }
}
void compute_grid_votes_per_pixel(double *image,int *votes,int X,int Y) {
 struct Context ctx={.image=image,.X=X,.Y=Y,.votes=votes};
 for(int k=0;k<8;k++)for(int l=0;l<8;l++)ctx.cos_t[k][l]=cos((2.0*k+1.0)*l*M_PI/16.0);
 ctx.zeros=xcalloc(X*Y,sizeof(unsigned char));ctx.constant=xcalloc(X*Y,sizeof(unsigned char));
 dispatch_queue_t queue=dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0);
 dispatch_apply_f(X-7,queue,&ctx,block_column);
 dispatch_apply_f(Y,queue,&ctx,vote_row);
 free(ctx.zeros);free(ctx.constant);
}
int zero(double * input, double * input_jpeg,
         double * luminance, double * luminance_jpeg,
         int * votes, int * votes_jpeg,
         double * lnfa_grids,
         meaningful_reg * foreign_regions, int * foreign_regions_n,
         meaningful_reg * missing_regions, int * missing_regions_n,
         int * mask_f, int * mask_f_reg, int * mask_m, int * mask_m_reg,
         int X, int Y, int C, int C_jpeg) {

    int main_grid = -1;

    /* luminance image */
    rgb2luminance(input, luminance, X, Y, C);

    /* compute vote map */
    compute_grid_votes_per_pixel(luminance, votes, X, Y);

    /* detect global grids and main_grid */
    main_grid = detect_global_grids(votes, lnfa_grids, X, Y);

    /* compute forged regions */
    *foreign_regions_n = detect_forgeries(votes, mask_f, mask_f_reg,
                                          foreign_regions, X, Y,
                                          main_grid, 63);

    /* if a global grid is found and the JPEG QF-99 version is provided,
       try to found regions with missing JPEG grid */
    if (main_grid > -1 && input_jpeg != NULL) {
        /* luminance image */
        rgb2luminance(input_jpeg, luminance_jpeg, X, Y, C_jpeg);

        /* compute vote map */
        compute_grid_votes_per_pixel(luminance_jpeg, votes_jpeg, X, Y);

        /* update votemap by avoiding the votes for the main grid */
        for (int x=0; x<X; x++)
            for (int y=0; y<Y; y++)
                if (votes[x+y*X] == main_grid)
                    votes_jpeg[x+y*X] = -1;

        /* Try to detect an imposed JPEG grid.  No grid is to be excluded
           and we are interested only in grid with origin (0,0), so:
           grid_to_exclude = -1 and grid_max = 0 */
        *missing_regions_n = detect_forgeries(votes_jpeg, mask_m, mask_m_reg,
                                              missing_regions, X, Y, -1, 0);
    }

    return main_grid;
}
/*----------------------------------------------------------------------------*/
