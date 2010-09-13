////////////////////////////////////
// Konrad Rodrigo Rodzik (c) 2010 //
////////////////////////////////////
#ifndef __H_CMail_H__
#define __H_CMail_H__

#include "jwsmtp/jwsmtp.h"

class CMail
{
public:
	// Initialize constructor
	CMail();

	// Set subject
	void setSubject(const char* title);

	// Set body
	void setBody(const char* body);

	// Add attachment
	void addAttachment(const char* attach_file);

	// Send e-mail
	const std::string send();

private:
	std::string m_smtpServer;	// Mail server
	std::string m_from;			// Sender mail
	std::string m_password;		// Sender password
	std::string m_to;			// Receiver mail
	std::string m_subject;		// Mail subject
	std::string m_body;			// Mail body

	jwsmtp::mailer m_mail;		// Mailer object
};

#endif