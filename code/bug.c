#include "mad_tpsa.h"

enum { NV = 4,
       MO = 5, 
};


int main(void)
{
  const desc_t *d = mad_desc_newv(NV, MO);

  tpsa_t* x = mad_tpsa_newd(d, 255);
  tpsa_t* y = mad_tpsa_newd(d, 255);
  tpsa_t* z = mad_tpsa_newd(d, 255);

  ord_t m[] = {5};

  mad_tpsa_seti(x, 0, 0.,1.);
  mad_tpsa_setm(y, 1, m, 0.,1.);

  mad_tpsa_add(x, y, z);

  mad_tpsa_print(z, "", 0., 0, 0);

  mad_tpsa_del(x);
  mad_tpsa_del(y);
  mad_tpsa_del(z);
  mad_desc_del(0);
  return 0;
}
